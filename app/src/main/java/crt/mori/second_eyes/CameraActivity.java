package crt.mori.second_eyes;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
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
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.core.Mat;
import org.opencv.features2d.Features2d;

import java.io.ByteArrayInputStream;
import java.util.List;
import java.util.Random;

import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.video.Video.calcOpticalFlowPyrLK;
import static org.opencv.video.Video.estimateRigidTransform;


public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private drawMap mSurfaceView;
    private Mat mRgbaFrame;
    private Mat mRgbaFramePrev;
    private Mat mGrayFrame;
    private Mat mGrayFramePrev;
    private Mat mOutFrame;
    private Mat mDescPointsPrev;
    private Scalar mKeyPointsColor = new Scalar(0.5,0.0,0.0,1.0);
    private Scalar mKeyPointsColorPrev = new Scalar(0.0,0.5,0.0,1.0);
    private FeatureDetector mFeatureDectector;
    private MatOfKeyPoint mKeyPoints;
    private MatOfKeyPoint mKeyPointsPrev;
    private MatOfPoint2f prevPoints2f;
    private DescriptorExtractor mDescExtractor;
    private DescriptorMatcher mDescMatcher;
    private MatOfDMatch mMatchPoints;
    private int mWidth;
    private int mHeight;
    boolean isRunning=false;
    float fPointX;
    float fPointY;

    Path mPath;

    Random mRandom;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mPath = new Path();

        mSurfaceView = new drawMap(this);
        LinearLayout layout = (LinearLayout) findViewById(R.id.MapDrawingLayout);
        mSurfaceView.setId(5000);
        mSurfaceView.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT, 0.4f));
        mSurfaceView.setBackgroundColor(Color.TRANSPARENT);
        mSurfaceView.setZOrderOnTop(true);
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
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        if (mSurfaceView != null) {
            mSurfaceView.pause();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        if (mSurfaceView != null) {
            mSurfaceView.pause();
        }
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
        mRgbaFramePrev = new Mat(height, width, CvType.CV_8UC4);
        mGrayFrame = new Mat(height, width, CvType.CV_8UC1);
        mGrayFramePrev = new Mat(height, width, CvType.CV_8UC1);
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

        if (!mGrayFramePrev.empty()) {
            Mat movement = estimateRigidTransform(mGrayFrame, mGrayFramePrev, false);
            if (!movement.empty()) {
                fPointX = (float) (movement.get(0, 2)[0]);
                fPointY = (float) (movement.get(1, 2)[0]);
                Log.i(TAG, "Adding to path (" + Float.toString(fPointX) + "," + Float.toString(fPointY) + ")");
                mPath.rLineTo(fPointX, fPointY);
            }
        }
        mGrayFrame.copyTo(mOutFrame);
        if (false) {
            Mat DescPoints = new Mat();
            MatOfDMatch matchingPoints = new MatOfDMatch();

            mFeatureDectector.detect(mRgbaFrame, mKeyPoints);
            mDescExtractor.compute(mRgbaFrame, mKeyPoints, DescPoints);

            DescPoints.copyTo(mDescPointsPrev);
            Features2d.drawKeypoints(mGrayFrame, mKeyPoints, mOutFrame, mKeyPointsColor, 0);

            // Convert MatOfKeyPoint to MatOfPoint2f
            KeyPoint[] kpoints = mKeyPoints.toArray();
            Point[] points = new Point[kpoints.length];
            for (int i = 0; i < kpoints.length; ++i) {
                points[i] = new Point(0, 0);
                points[i].x = kpoints[i].pt.x;
                points[i].y = kpoints[i].pt.y;
            }
            MatOfPoint2f currPoints2f = new MatOfPoint2f(points);

            if (prevPoints2f != null) {
                //Features2d.drawKeypoints(mGrayFrame, mKeyPointsPrev, mOutFrame, mKeyPointsColor, 0);

            /*MatOfByte matches = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            calcOpticalFlowPyrLK(mGrayFramePrev,mGrayFrame,prevPoints2f, currPoints2f, matches, err);

            // remove points where LK tracking failed or that went missing
            for(int i=0; i < matches.toArray().length; ++i)
            {
                Point pt = currPoints2f.toArray()[i];
                if ((matches.toArray()[i] == 0)||(pt.x<0)||(pt.y<0))	{
                    currPoints2f.toList().remove(pt);
                }
            }
*/
                // now lets get out rigidTransformation for orientation vector
                try {
                    Mat movement = findHomography(currPoints2f, prevPoints2f, Calib3d.LMEDS, 1.2f);
                    Log.i(TAG, "We have points. x=" + movement.get(0, 2).toString() + "y=" + movement.get(1, 2).toString());

                    fPointX = (float) (movement.get(0, 2)[0]) * 10000000000000.0f;
                    fPointY = (float) (movement.get(1, 2)[0]) * 10000000000000.0f;
                    Log.i(TAG, "Adding to path (" + Float.toString(fPointX) + "," + Float.toString(fPointY) + ")");
                    mPath.rLineTo(fPointX, fPointY);
                } catch (CvException e) {
                    Log.i(TAG, "Vectors are the same");
                    //e.printStackTrace();
                }

                isRunning = true;
                //mSurfaceView.invalidate();

            }
            mKeyPoints.copyTo(mKeyPointsPrev);
            prevPoints2f = new MatOfPoint2f(points);
        }

        mGrayFrame.copyTo(mGrayFramePrev);
        mRgbaFrame.copyTo(mRgbaFramePrev);

        return mOutFrame;
    }

    public class drawMap extends SurfaceView implements Runnable {
        private static final String TAG = "drawMap";
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
            Log.i(TAG, "Generating thread");
            isRunning = true;
            mThread = new Thread(this);
            mThread.start();
        }

        @Override
        public void run() {
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(20);
            paint.setColor(Color.WHITE);
            Canvas canvas;

            Log.i(TAG, "run() Init");
            while(true) {
                if(mHolder.getSurface().isValid()) {
                    canvas = mHolder.lockCanvas();
                    // set Path starting point
                    mPath.moveTo(canvas.getWidth() / 2.0f, canvas.getHeight() / 2.0f);
                    mHolder.unlockCanvasAndPost(canvas);
                    break;
                }
                mThread.yield();
            }

            Log.i(TAG, "run() Running started");
            while(isRunning) {
                if (mHolder.getSurface().isValid()) {
                    canvas = mHolder.lockCanvas();
                    canvas.drawPath(mPath, paint);
                    mHolder.unlockCanvasAndPost(canvas);
                }
                mThread.yield();
            }
        }

    }

}
