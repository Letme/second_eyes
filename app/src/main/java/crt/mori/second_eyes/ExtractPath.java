package crt.mori.second_eyes;

import android.graphics.Path;
import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvException;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.video.Video.estimateRigidTransform;

/**
 * Various methods of extracting path from images, matrices, etc. It is a simplified wrapper for the
 * OpenCV library as each function only returns a path (which can be drawn) of the differences
 * between two images/matrices, and takes in various modifiers it need to calculate that difference
 * with different methods.
 */
public class ExtractPath {

    private static final String TAG="ExtractPath";

    private FeatureDetector mFeatureDectector;
    private DescriptorExtractor mDescExtractor;
    private DescriptorMatcher mDescMatcher;
    private MatOfKeyPoint mKeyPointsPrev;
    private MatOfPoint2f prevPoints2f;
    private Scalar mKeyPointsColor = new Scalar(0.5,0.0,0.0,1.0);
    private List<MatOfPoint> mContours;
    private Mat mPrevFrame;
    private Mat RGBFrame;
    private Mat mForeGroundMask;
    private MatOfKeyPoint prevKeyPoints;
    private double mRefreshRate = 0.5;
    private BackgroundSubtractor mBackgroundSub;


    public ExtractPath() {
        super();

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

        mPrevFrame = new Mat();
        prevKeyPoints = new MatOfKeyPoint();
        RGBFrame = new Mat();
        mForeGroundMask = new Mat();
        mContours = new ArrayList<MatOfPoint>();
        //creates a new BackgroundSubtractorMOG class with the arguments
        mBackgroundSub = Video.createBackgroundSubtractorMOG2(50, 0, true);
    }

    /**
     * Extract object movement from current and previous frame using Rigid Transformation and save
     * it to path passed to function.
     *
     * @param currFrame Current frame
     * @param prevFrame Previous frame
     * @param path Path variable to which object movement is saved
     * @return Current frame for displaying
     */
    public Path withRigidTransformation(Mat currFrame, Mat prevFrame, Path path) {
        Mat movement = estimateRigidTransform(currFrame, prevFrame, false);
        if (!movement.empty()) {
            float fX = (float) (movement.get(0, 2)[0]);
            float fY = (float) (movement.get(1, 2)[0]);
            Log.i(TAG, "Adding to path (" + Float.toString(fX) + "," + Float.toString(fY) + ")");
            path.rLineTo(fX, fY);
        }
        return path;
    }

    public Path withContours(Mat grayFrame, Path path) {
        // Convert gray frame to RGB frame for background removal
        Imgproc.cvtColor(grayFrame, RGBFrame, Imgproc.COLOR_GRAY2RGB);

        mBackgroundSub.apply(RGBFrame, mForeGroundMask, mRefreshRate);

        Imgproc.erode(mForeGroundMask, mForeGroundMask, new Mat());
        Imgproc.dilate(mForeGroundMask, mForeGroundMask, new Mat());

        Imgproc.findContours(mForeGroundMask, mContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        //draws all the contours in red with thickness of 2
        Imgproc.drawContours(RGBFrame, mContours, -1, new Scalar(255, 0, 0), 2);
        RGBFrame.copyTo(grayFrame);
        return path;
    }

    public Path withFAST(Mat currFrame, Mat mOutFrame, Path path) {
        MatOfKeyPoint currKeyPoints = new MatOfKeyPoint();
        float fPointX, fPointY;

        mFeatureDectector.detect(currFrame, currKeyPoints);

        Features2d.drawKeypoints(currFrame, currKeyPoints, mOutFrame, mKeyPointsColor, 0);

        if (mPrevFrame != null) {
            // now lets get out rigidTransformation for orientation vector
           try {
               Mat movement = estimateRigidTransform(mOutFrame, mPrevFrame, false);
                //Mat movement = findHomography(currPoints2f, prevPoints2f, Calib3d.LMEDS, 1.2f);
               if (!movement.empty()) {
                   fPointX = (float) (movement.get(0, 2)[0]);
                   fPointY = (float) (movement.get(1, 2)[0]);
                   Log.i(TAG, "Adding to path (" + Float.toString(fPointX) + "," + Float.toString(fPointY) + ")");
                   path.rLineTo(fPointX, fPointY);
               }
            } catch (CvException e) {
                Log.i(TAG, "Vectors are the same");
                //e.printStackTrace();
            }
        }
        mOutFrame.copyTo(mPrevFrame);

        return path;
    }
}
