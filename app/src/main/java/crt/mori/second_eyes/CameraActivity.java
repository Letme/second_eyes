package crt.mori.second_eyes;

import android.annotation.SuppressLint;
import android.graphics.Color;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.MenuItem;
import android.support.v4.app.NavUtils;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.core.Mat;
import org.opencv.features2d.Features2d;


public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgbaFrame;
    private Mat mGrayFrame;
    private Mat mOutFrame;
    private Mat mDescPointsPrev;
    private Scalar mKeyPointsColor = new Scalar(0.5,0.5,0.0,1.0);
    private FeatureDetector mFeatureDectector;
    private MatOfKeyPoint mKeyPoints;
    private DescriptorExtractor mDescExtractor;
    private DescriptorMatcher mDescMatcher;
    private MatOfDMatch mMatchPoints;
    private int mWidth;
    private int mHeight;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_camera);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == android.R.id.home) {
            // This ID represents the Home or Up button.
            NavUtils.navigateUpFromSameTask(this);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public void onCameraViewStarted(int width, int height) {
        // set up start variables
        mRgbaFrame = new Mat(height, width, CvType.CV_8UC4);
        mGrayFrame = new Mat(height, width, CvType.CV_8UC1);
        mOutFrame = new Mat(height, width, CvType.CV_8UC4);
        mDescPointsPrev = new Mat(height, width, CvType.CV_8UC4);
        mKeyPoints = new MatOfKeyPoint();

        // set up feature detection
        try {
            mFeatureDectector = FeatureDetector.create(FeatureDetector.FAST);
        } catch (UnsatisfiedLinkError err) {
            Log.e(TAG, "Feature detector failed with");
            err.printStackTrace();
        }
        // set up description detection
        mDescExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        mDescMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);




    }

    public void onCameraViewStopped() {
        mRgbaFrame.release();
        mGrayFrame.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgbaFrame = inputFrame.rgba();
        mGrayFrame = inputFrame.gray();

        Mat DescPoints = new Mat();
        MatOfDMatch matchingPoints = new MatOfDMatch();

        mFeatureDectector.detect(mRgbaFrame, mKeyPoints);
        mDescExtractor.compute(mRgbaFrame, mKeyPoints, DescPoints);

        if (!mDescPointsPrev.empty() && !DescPoints.empty()) {
            //mDescMatcher.match(DescPoints, mDescPointsPrev, matchingPoints);
        }
        DescPoints.copyTo(mDescPointsPrev);
        Features2d.drawKeypoints(mGrayFrame,mKeyPoints,mOutFrame, mKeyPointsColor,0);
        return mOutFrame;
    }
}
