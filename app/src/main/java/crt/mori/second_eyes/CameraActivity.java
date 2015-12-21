package crt.mori.second_eyes;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.MenuItem;
import android.support.v4.app.NavUtils;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.MultiAutoCompleteTextView;

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

import java.util.Random;


public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private SurfaceView mSurfaceView;
    private Mat mRgbaFrame;
    private Mat mGrayFrame;
    private Mat mOutFrame;
    private Mat mDescPointsPrev;
    private Scalar mKeyPointsColor = new Scalar(0.5,0.0,0.0,1.0);
    private Scalar mKeyPointsColorPrev = new Scalar(0.0,0.5,0.0,1.0);
    private FeatureDetector mFeatureDectector;
    private MatOfKeyPoint mKeyPoints;
    private MatOfKeyPoint mKeyPointsPrev;
    private DescriptorExtractor mDescExtractor;
    private DescriptorMatcher mDescMatcher;
    private MatOfDMatch mMatchPoints;
    private int mWidth;
    private int mHeight;
    boolean isRunning=false;
    int PointX = 0;
    int PointY = 0;

    Random mRandom;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mSurfaceView = new drawMap(this);
        LinearLayout layout = (LinearLayout) findViewById(R.id.MapDrawingLayout);
        mSurfaceView.setId(5000);
        mSurfaceView.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT, 0.4f));
        mSurfaceView.setBackgroundColor(Color.BLUE);
        layout.addView(mSurfaceView);
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
        mKeyPointsPrev = new MatOfKeyPoint();

        // set up feature detection
        try {
            mFeatureDectector = FeatureDetector.create(FeatureDetector.FAST);
        } catch (UnsatisfiedLinkError err) {
            Log.e(TAG, "Feature detector failed with");
            err.printStackTrace();
        }
        // set up description detection
        mDescExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
        mDescMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        mRandom = new Random();


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

        DescPoints.copyTo(mDescPointsPrev);
        Features2d.drawKeypoints(mGrayFrame, mKeyPoints, mOutFrame, mKeyPointsColor, 0);
        if (!mKeyPointsPrev.empty()) {
            Features2d.drawKeypoints(mGrayFrame, mKeyPointsPrev, mOutFrame, mKeyPointsColor, 0);
            // zmanjsas matrko mKeyPointsPrev in jo primerjas z trenutno matrko tako da isces prejsnjo matriko
            // znotraj trenutne. Uporabis lahko bruteforce matcher.
            mDescMatcher.match(DescPoints,mDescPointsPrev.adjustROI(8,8,8,8), matchingPoints);

            PointX = mRandom.nextInt(100);
            PointY = mRandom.nextInt(100);

            isRunning = true;
            //mSurfaceView.invalidate();

        }
        mKeyPoints.copyTo(mKeyPointsPrev);

        return mOutFrame;
    }

    public class drawMap extends SurfaceView implements Runnable {
        private static final String TAG = "drawMap";
        private Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        SurfaceHolder mHolder;
        Thread mThread = null;

        public drawMap(Context context) {
            super(context);

            Log.i(TAG, "Starting...");
            mHolder = getHolder();
            setWillNotDraw(false);
            onSurfaceCreated();
        }

        public void pause() {
            isRunning = false;
            while(true) {
                try {
                    mThread.join();
                }
                catch(InterruptedException e) {
                    e.printStackTrace();
                }
                break;
            }
            mThread = null;
            //mThread.destroy();
        }

        public void onSurfaceCreated() {
            Log.i(TAG, "Gnerating thread");
            isRunning = true;
            mThread = new Thread(this);
            mThread.start();
        }

        @Override
        public void run() {
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(20);
            paint.setColor(Color.WHITE);

            Log.i(TAG, "------- Run ----------");
            while(isRunning) {
                if (mHolder.getSurface().isValid()) {
                    if ((PointX != 0) && (PointY != 0)) {
                        Canvas canvas = mHolder.lockCanvas();
                        Log.i(TAG,"Drawing (" + Integer.toString(PointX) + "," + Integer.toString(PointY) + ")");
                        canvas.drawPoint(PointX, PointY, paint);
                        mHolder.unlockCanvasAndPost(canvas);
                    }
                }
                //mThread.yield();
            }
        }

    }

}
