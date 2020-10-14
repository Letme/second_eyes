package crt.mori.second_eyes;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.MenuItem;
import androidx.core.app.NavUtils;

import android.view.WindowManager;
import android.widget.LinearLayout;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.Mat;

import java.util.Random;

import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.video.Video.calcOpticalFlowPyrLK;

public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private enum mTransformation { RIGID, FAST, CONTOURS };
    private mTransformation mActiveTransformation = mTransformation.RIGID;

    private CameraBridgeViewBase mOpenCvCameraView;
    private drawMap mSurfaceView;
    private Mat mRgbaFrame;
    private Mat mRgbaFramePrev;
    private Mat mGrayFrame;
    private Mat mGrayFramePrev;
    private Mat mOutFrame;
    private Mat mDescPointsPrev;

    private MatOfDMatch mMatchPoints;
    private int mWidth;
    private int mHeight;
    boolean isRunning=false;
    float fPointX;
    float fPointY;
    private ExtractPath mExtractPath;

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
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.camera, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        boolean retval= false;
        int id = item.getItemId();
        switch (id) {
            case android.R.id.home:
                // This ID represents the Home or Up button.
                NavUtils.navigateUpFromSameTask(this);
                retval = true;
                break;
            case R.id.action_rigid:
                mActiveTransformation = mTransformation.RIGID;
                retval = true;
                break;
            case R.id.action_fast_homography:
                mActiveTransformation = mTransformation.FAST;
                retval = true;
                break;
            case R.id.action_contours:
                mActiveTransformation = mTransformation.CONTOURS;
                retval = true;
                break;
            default:
                return super.onOptionsItemSelected(item);

        }

        return retval;
    }

    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
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

    @Override
    public void onCameraViewStarted(int width, int height) {
        // set up start variables
        mRgbaFrame = new Mat(height, width, CvType.CV_8UC4);
        mRgbaFramePrev = new Mat(height, width, CvType.CV_8UC4);
        mGrayFrame = new Mat(height, width, CvType.CV_8UC1);
        mGrayFramePrev = new Mat(height, width, CvType.CV_8UC1);
        mOutFrame = new Mat(height, width, CvType.CV_8UC4);
        mDescPointsPrev = new Mat(height, width, CvType.CV_8UC4);

        mExtractPath = new ExtractPath();

        mRandom = new Random();
    }

    @Override
    public void onCameraViewStopped() {
        mRgbaFrame.release();
        mGrayFrame.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgbaFrame = inputFrame.rgba();
        mGrayFrame = inputFrame.gray();

        if (!mGrayFramePrev.empty()) {
            switch (mActiveTransformation) {
                case RIGID:
                    mPath = mExtractPath.withRigidTransformation(mGrayFrame, mGrayFramePrev, mPath);
                    mGrayFrame.copyTo(mOutFrame);
                    break;
                case FAST:
                    mPath = mExtractPath.withFAST(mGrayFrame, mOutFrame, mPath);
                    //mGrayFrame.copyTo(mOutFrame);
                    break;
                case CONTOURS:
                    mPath = mExtractPath.withContours(mGrayFrame, mPath);
                    //mGrayFrame.copyTo(mOutFrame);
                    break;
                default:
                    break;
            }
        } else {
            mOutFrame = inputFrame.gray();
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
            Log.i(TAG, "Ending drawing thread");
            isRunning = false;
            while (mThread != null) {
                mThread.interrupt();
                mThread = null;
            }
        }

        public void onSurfaceCreated() {
            Log.i(TAG, "Starting drawing thread");
            isRunning = true;
            mThread = new Thread(this);
            mThread.start();
        }

        @Override
        public void run() {
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(20);
            switch (mActiveTransformation) {
                case RIGID:
                    paint.setColor(Color.WHITE);
                    break;
                case FAST:
                    paint.setColor(Color.BLUE);
                    break;
                case CONTOURS:
                    paint.setColor(Color.RED);
                    break;
                default:
                    paint.setColor(Color.GREEN);
                    break;
            }

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

            Log.i(TAG, "run() Ended");
        }

    }

}
