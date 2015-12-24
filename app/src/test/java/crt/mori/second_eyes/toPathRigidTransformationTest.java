package crt.mori.second_eyes;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Path;
import android.test.AssertionFailedError;
import android.test.InstrumentationTestCase;
import android.util.Log;


import junit.framework.Assert;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class toPathRigidTransformationTest {

    private void setUp() {
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {

                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Test
    public void object_to_left() {
        Assert.assertTrue(true);
        ExtractPath mExtractPath = new ExtractPath();

        Path myPath = new Path();
        Bitmap bmp = null;

        Mat currFrame = new Mat();
        Utils.bitmapToMat(bmp,currFrame);

        Mat prevFrame = new Mat();
        Utils.bitmapToMat(bmp,currFrame);
        Assert.assertEquals(currFrame, prevFrame);
        Assert.assertSame(currFrame, mExtractPath.withRigidTransformation(currFrame, prevFrame, myPath));

    }
}