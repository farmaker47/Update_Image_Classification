/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;

import androidx.appcompat.app.AppCompatActivity;

import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.UiThread;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.databinding.DataBindingUtil;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.tensorflow.lite.examples.classification.databinding.TfeIcActivityCameraBinding;

import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Recognition;

public class CameraActivity extends AppCompatActivity
        implements View.OnClickListener,
        AdapterView.OnItemSelectedListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

  private Handler handler;
  private HandlerThread handlerThread;
  private boolean firstTimeStartModel = true;
  private boolean isProcessingFrame = false;

  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private Model model = Model.QUANTIZED_EFFICIENTNET;
  private Device device = Device.CPU;
  private int numThreads = -1;

  public static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  /**
   * Input image size of the model along x axis.
   */
  private int imageSizeX;
  /**
   * Input image size of the model along y axis.
   */
  private int imageSizeY;

  private TfeIcActivityCameraBinding binding;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    binding = DataBindingUtil.setContentView(this, R.layout.tfe_ic_activity_camera);

    if (hasPermission()) {
      // Start CameraX
      startCamera();
    } else {
      requestPermission();
    }

    LinearLayout bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);

    ViewTreeObserver vto = binding.bottomSheetLayout.gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                binding.bottomSheetLayout.gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int height = binding.bottomSheetLayout.gestureLayout.getMeasuredHeight();
                sheetBehavior.setPeekHeight(height);
              }
            });
    sheetBehavior.setHideable(false);

    sheetBehavior.addBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED: {
                    binding.bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED: {
                    binding.bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_DRAGGING:
                    break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    binding.bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }

              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {
              }
            });

    binding.bottomSheetLayout.modelSpinner.setOnItemSelectedListener(this);
    binding.bottomSheetLayout.deviceSpinner.setOnItemSelectedListener(this);

    binding.bottomSheetLayout.plus.setOnClickListener(this);
    binding.bottomSheetLayout.minus.setOnClickListener(this);

    model = Model.valueOf(binding.bottomSheetLayout.modelSpinner.getSelectedItem().toString().toUpperCase());
    device = Device.valueOf(binding.bottomSheetLayout.deviceSpinner.getSelectedItem().toString());
    numThreads = Integer.parseInt(binding.bottomSheetLayout.threads.getText().toString().trim());
  }

  @SuppressLint("UnsafeOptInUsageError")
  private void startCamera() {
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

    cameraProviderFuture.addListener(() -> {
      // Camera provider is now guaranteed to be available
      try {
        ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

        // Set up the view finder use case to display camera preview
        Preview preview = new Preview.Builder().build();

        // Choose the camera by requiring a lens facing
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        // Image Analysis
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(DESIRED_PREVIEW_SIZE)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), image -> {
          // Define rotation Degrees of the imageProxy
          int rotationDegrees = image.getImageInfo().getRotationDegrees();
          Log.v("ImageAnalysis_degrees", String.valueOf(rotationDegrees));

          // Execute this method to start the model ONCE
          if (firstTimeStartModel) {
            onStartCameraX(DESIRED_PREVIEW_SIZE, rotationDegrees);
            firstTimeStartModel = false;
          }

          if (!isProcessingFrame) {
            final int cropSize = Math.min(DESIRED_PREVIEW_SIZE.getWidth(), DESIRED_PREVIEW_SIZE.getHeight());

            runInBackground(
                    () -> {
                      if (classifier != null) {
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = classifier.recognizeImage(image.getImage(), sensorOrientation);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.e("Degrees: %s", results);

                        runOnUiThread(
                                () -> {
                                  showResultsInBottomSheet(results);
                                  showFrameInfo(DESIRED_PREVIEW_SIZE.getWidth() + "x" + DESIRED_PREVIEW_SIZE.getHeight());
                                  showCropInfo(imageSizeX + "x" + imageSizeY);
                                  showCameraResolution(cropSize + "x" + cropSize);
                                  showRotationInfo(String.valueOf(sensorOrientation));
                                  showInference(lastProcessingTimeMs + "ms");
                                });
                      }
                      image.close();
                      isProcessingFrame = false;
                    });

            isProcessingFrame = true;
          }

        });

        // Connect the preview use case to the previewView
        preview.setSurfaceProvider(
                binding.previewView.getSurfaceProvider());

        // Attach use cases to the camera with the same lifecycle owner
        if (cameraProvider != null) {
          Camera camera = cameraProvider.bindToLifecycle(
                  this,
                  cameraSelector,
                  imageAnalysis,
                  preview);
        }

      } catch (ExecutionException | InterruptedException e) {
        e.printStackTrace();
      }


    }, ContextCompat.getMainExecutor(this));
  }

  private void onStartCameraX(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    BorderedText borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    // Use rotation which is the rotation of the Image that is provided by the CameraX
    sensorOrientation = rotation;
    LOGGER.i("Camera orientation sensor relative to screen canvas: %d", sensorOrientation);
    LOGGER.i("Camera orientation screen relative to screen canvas: %d", getScreenOrientation());

    LOGGER.i("Initializing at size %dx%d", DESIRED_PREVIEW_SIZE.getWidth(), DESIRED_PREVIEW_SIZE.getHeight());
  }

  protected void onInferenceConfigurationChanged() {
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU && (model == Model.QUANTIZED_MOBILENET || model == Model.QUANTIZED_EFFICIENTNET)) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
              () -> Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show());
      return;
    }
    try {
      LOGGER.d("Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException | IllegalArgumentException e) {
      LOGGER.e(e, "Failed to create classifier.");
      runOnUiThread(
              () -> Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show());
      return;
    }

    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        // Start CameraX
        startCamera();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
                .show();
      }
      requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  protected int getScreenOrientation() {
    switch (this.getDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @UiThread
  protected void showResultsInBottomSheet(List<Recognition> results) {
    if (results != null && results.size() >= 3) {
      Recognition recognition = results.get(0);
      if (recognition != null) {
        if (recognition.getTitle() != null)
          binding.bottomSheetLayout.detectedItem.setText(recognition.getTitle());
        if (recognition.getConfidence() != null)
          binding.bottomSheetLayout.detectedItemValue.setText(
                  String.format("%.2f", (100 * recognition.getConfidence())) + "%");
      }

      Recognition recognition1 = results.get(1);
      if (recognition1 != null) {
        if (recognition1.getTitle() != null)
          binding.bottomSheetLayout.detectedItem1.setText(recognition1.getTitle());
        if (recognition1.getConfidence() != null)
          binding.bottomSheetLayout.detectedItem1Value.setText(
                  String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
      }

      Recognition recognition2 = results.get(2);
      if (recognition2 != null) {
        if (recognition2.getTitle() != null)
          binding.bottomSheetLayout.detectedItem2.setText(recognition2.getTitle());
        if (recognition2.getConfidence() != null)
          binding.bottomSheetLayout.detectedItem2Value.setText(
                  String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
      }
    }
  }

  protected void showFrameInfo(String frameInfo) {
    binding.bottomSheetLayout.frameInfo.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    binding.bottomSheetLayout.cropInfo.setText(cropInfo);
  }

  protected void showCameraResolution(String cameraInfo) {
    binding.bottomSheetLayout.viewInfo.setText(cameraInfo);
  }

  protected void showRotationInfo(String rotation) {
    binding.bottomSheetLayout.rotationInfo.setText(rotation);
  }

  protected void showInference(String inferenceTime) {
    binding.bottomSheetLayout.inferenceInfo.setText(inferenceTime);
  }

  protected Model getModel() {
    return model;
  }

  private void setModel(Model model) {
    if (this.model != model) {
      LOGGER.d("Updating  model: " + model);
      this.model = model;
      onInferenceConfigurationChanged();
    }
  }

  protected Device getDevice() {
    return device;
  }

  private void setDevice(Device device) {
    if (this.device != device) {
      LOGGER.d("Updating  device: " + device);
      this.device = device;
      final boolean threadsEnabled = device == Device.CPU;
      binding.bottomSheetLayout.plus.setEnabled(threadsEnabled);
      binding.bottomSheetLayout.minus.setEnabled(threadsEnabled);
      binding.bottomSheetLayout.threads.setText(threadsEnabled ? String.valueOf(numThreads) : "N/A");
      onInferenceConfigurationChanged();
    }
  }

  protected int getNumThreads() {
    return numThreads;
  }

  private void setNumThreads(int numThreads) {
    if (this.numThreads != numThreads) {
      LOGGER.d("Updating  numThreads: " + numThreads);
      this.numThreads = numThreads;
      onInferenceConfigurationChanged();
    }
  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      setNumThreads(++numThreads);
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
    } else if (v.getId() == R.id.minus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      setNumThreads(--numThreads);
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
    }
  }

  @Override
  public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
    if (parent == binding.bottomSheetLayout.modelSpinner) {
      setModel(Model.valueOf(parent.getItemAtPosition(pos).toString().toUpperCase()));
    } else if (parent == binding.bottomSheetLayout.deviceSpinner) {
      setDevice(Device.valueOf(parent.getItemAtPosition(pos).toString()));
    }
  }

  @Override
  public void onNothingSelected(AdapterView<?> parent) {
    // Do nothing.
  }
}
