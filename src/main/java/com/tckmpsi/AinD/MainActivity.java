package com.tckmpsi.AinD;

import static android.util.Base64.DEFAULT;
import static android.util.Base64.encodeToString;
import static java.util.Base64.*;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.pdf.PdfDocument;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.w3c.dom.Document;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Base64;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;

    // declaring width and height
    // for our PDF file.
    int pageHeight = 550;
    int pagewidth = 650;
    public String ImagefileName=null;
    // constant code for runtime permissions
    private static final int PERMISSION_REQUEST_CODE = 200;

    Bitmap  scaledbmp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
         //String myimageString;

        TextView textView = findViewById(R.id.result_text);
        TextView textView2 = findViewById(R.id.result_text2);
androidx.appcompat.app.ActionBar actionBar=getSupportActionBar();
//actionBar.setBackgroundDrawable(getResources().getDrawable(R.drawable.hbku_logo));
getSupportActionBar().setDisplayShowTitleEnabled(true);

        actionBar.setDisplayShowCustomEnabled(true);
        LayoutInflater  inflater=(LayoutInflater)this.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View view=inflater.inflate(R.layout.custom_image,null);

        actionBar.setCustomView(view);


        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);
        //Button camera = findViewById(R.id.button2);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        // below code is used for
        // checking our permissions.
        if (checkPermission()) {
            Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
        } else {

            requestPermission();
        }

       /* camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    TextView textView = findViewById(R.id.result_text);
                    textView.setText("");
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });*/
        buttonLoadImage.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                TextView textView2 = findViewById(R.id.result_text2);
                TextView textView3 = findViewById(R.id.result_text3);
                textView.setText("---");
                textView2.setText("---");
                textView3.setText("---");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i, RESULT_LOAD_IMAGE);


            }
        });
        if(!Python.isStarted())
            Python.start(new AndroidPlatform(this));
        final Python py=Python.getInstance();



        detectButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {

                Bitmap bitmap = null;
                Module module = null;
                Bitmap bmp=null;
                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);

                try {
                    //Read the image as Bitmap
                    bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                    //Here we reshape the image into 400*400
                    //bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);
                    //bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
                    //imageString=getStringImage(bitmap);
                    final String myimageString =getStringImage2(bitmap);
                    PyObject pyo=py.getModule("myscriptt");
                    //PyObject obj=pyo.callAttr("main");
                      PyObject obj=pyo.callAttr("main",myimageString);
                    String str=obj.toString();
                    //textView.setText(str);


                    //byte  data[] =android.util.Base64.decode(str, DEFAULT);
                    byte  data[]  = android.util.Base64.decode(str,DEFAULT);
                    bmp=BitmapFactory.decodeByteArray(data,0,data.length);
                    bmp = Bitmap.createScaledBitmap(bmp, 224, 224, true);
                    /////imageView.setImageBitmap(bmp);

                    //Loading the model file.
                    module = Module.load(fetchModelFile(MainActivity.this, "mobilemodel.pt"));


                } catch (IOException e) {
                    finish();
                }

                //Input Tensor
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bmp, //to pass mean subtracted image enable this
                        //bitmap,//to pass mean subtracted image comment this and uncoment above bmp
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );

                //Calling the forward of the model to run our input
                final Tensor output = module.forward(IValue.from(input)).toTensor();


                final float[] score_arr = output.getDataAsFloatArray();

                // Fetch the index of the value with maximum score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                double total = 0;
                String perf="";
                String perf1="";
                String perf2="";
                String perf_pdf="";
                for (int i = 0; i < score_arr.length; i++) {
                    if (score_arr[i] > max_score) {
                        max_score = score_arr[i];
                        ms_ix = i;
                        double value = Math.exp(score_arr[i]);
                        total += value;
                    }
                }

                //Fetching the name from the list based on the index
                String detected_class_topdf = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[ms_ix];
                String detected_class="";
                String detected_class1="";
                String detected_class2="";
                double totalvalue = (Math.exp(score_arr[0])) +(Math.exp(score_arr[1])) +(Math.exp(score_arr[2]));
                double myresult0 = (Math.exp(score_arr[0]) / totalvalue)*100;
                double myresult1 = (Math.exp(score_arr[1]) / totalvalue)*100;
                double myresult2 = (Math.exp(score_arr[2]) / totalvalue)*100;
                //double myresult0 = Math.exp(score_arr[0]);
                //double myresult1 = Math.exp(score_arr[1]);
                //String perf=String.format("total: %s, result0: %s,result1: %s", total, myresult0,myresult1);
                if(ms_ix==0){
                    perf_pdf=String.format(" %.1f%%%n", myresult0);
//                     detected_class = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[0];
//                    perf2=String.format(" %.1f%%%n", 100-myresult0);
                }
                else if(ms_ix==1){
                    perf_pdf=String.format(" %.1f%%%n", myresult1);
//                    detected_class1 = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[1];
//                    perf2=String.format(" %.1f%%%n", 100-myresult1);
                }
                else if(ms_ix==2){
                    perf_pdf=String.format(" %.1f%%%n", myresult2);
//                     detected_class2 = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[2];
//                    perf2=String.format(" %.1f%%%n", 100-myresult1);
                }

                //perf=String.format(" softmax[0]: %s,softmax[1]: %s",  myresult0,myresult1);
                //Writing the detected class in to the text view of the layout

                detected_class = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[0];
                detected_class1 = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[1];
                detected_class2 = com.tckmpsi.AinD.ModelClasses.MODEL_CLASSES[2];

                perf=String.format(" %.1f%%%n", myresult0);
                perf1=String.format(" %.1f%%%n", myresult1);
                perf2=String.format(" %.1f%%%n", myresult2);

                TextView textView = findViewById(R.id.result_text);
                TextView textView2 = findViewById(R.id.result_text2);
                TextView textView3 = findViewById(R.id.result_text3);
                textView.setText(detected_class+"  "+perf);
                textView2.setText(detected_class1+"  "+perf1);
                textView3.setText(detected_class2+"  "+perf2);
                //textView.setText(ms_ix+detected_class+" ([ "+score_arr[0]+" :"+score_arr[1]+"])  "+perf);
                //createPdf(detected_class+"  "+perf.toString());
                String class1=detected_class_topdf+"  "+perf_pdf;
                //String class2=detected_class2+"  "+perf2;

                String imagename=String.valueOf(imageView.getTag());
                Bitmap hbku_logo = BitmapFactory.decodeResource(getResources(),R.drawable.hbku_logo);
                generatePDF(ImagefileName,class1, hbku_logo,bitmap,bmp);

            }
        });

    }

    private String getStringImage2(Bitmap bmp) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.PNG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        //String encodedImage = encodeToString(imageBytes, DEFAULT);
        String encodedImage = android.util.Base64.encodeToString(imageBytes,DEFAULT);
        return encodedImage;

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                /*Uri selectedImage = data.getData();
                Cursor cursor = getContentResolver().query(selectedImage,
                        filePathColumn, null, null, null);
                cursor.moveToFirst();*/

                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);

                ImageView imageView = (ImageView) findViewById(R.id.image);
                imageView.setImageBitmap(image);

                //Setting the URI so we can read the Bitmap from the image
                //imageView.setImageURI(null);
                //imageView.setImageURI(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                ImageView imageView = (ImageView) findViewById(R.id.image);
                imageView.setImageBitmap(image);


            }
        }
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);

            String myImagefileName = picturePath.substring(0,picturePath.lastIndexOf("."));
            File file = new File(myImagefileName);
            ImagefileName = file.getName();

        }


    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    public void createpdf1(String bmp)throws FileNotFoundException{
  String pdfpath= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).toString();
  File file=new File(pdfpath, "mypdf.pdf");
  OutputStream  outputStream=new FileOutputStream(file);



    }



////////////////////////////////////////////////////////////---------------------------------------------------------
    private void generatePDF(String ImagefileName,String class1,Bitmap hbkulogo, Bitmap bitmaporigi,  Bitmap MyBitmap1) {
        // creating an object variable
        // for our PDF document.

        PdfDocument pdfDocument = new PdfDocument();
        //PdfDocument document = new PdfDocument();
        // crate a page description
        // two variables for paint "paint" is used
        // for drawing shapes and we will use "title"
        // for adding text in our PDF file.
        Paint paint = new Paint();
        Paint title = new Paint();

        // we are adding page info to our PDF file
        // in which we will be passing our pageWidth,
        // pageHeight and number of pages and after that
        // we are calling it to create our PDF.
        PdfDocument.PageInfo mypageInfo = new PdfDocument.PageInfo.Builder(pagewidth, pageHeight, 1).create();

        // below line is used for setting
        // start page for our PDF file.
        PdfDocument.Page myPage = pdfDocument.startPage(mypageInfo);

        // creating a variable for canvas
        // from our page of PDF.
        Canvas canvas = myPage.getCanvas();

        // below line is used to draw our image on our PDF file.
        // the first parameter of our drawbitmap method is
        // our bitmap
        // second parameter is position from left
        // third parameter is position from top and last
        // one is our variable for paint.
        /////////////////////////////canvas.drawBitmap(scaledbmp, 56, 40, paint);

        // below line is used for adding typeface for
        // our text which we will be adding in our PDF file.
        title.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.NORMAL));

        // below line is used for setting text size
        // which we will be displaying in our PDF file.
        title.setTextSize(15);

        // below line is sued for setting color
        // of our text inside our PDF file.
        title.setColor(ContextCompat.getColor(this, R.color.colorPrimaryDark));

        // below line is used to draw text in our PDF file.
        // the first parameter is our text, second parameter
        // is position from start, third parameter is position from top
        // and then we are passing our variable of paint which is title.
        Rect source = new Rect(0, 0, hbkulogo.getWidth(), hbkulogo.getHeight());
        Rect bitmapRect = new Rect(0, 0, 200,50);
        canvas.drawBitmap(hbkulogo, source, bitmapRect, null);

        canvas.drawText("Ain-DME Report", 209, 70, title);
        canvas.drawText("Patient No: "+ImagefileName, 209, 90, title);  
        canvas.drawText("Results: "+class1, 20, 120, title);
        //canvas.drawText("Results: "+class2, 20, 140, title);

        //example
        /*resize bitmap*/
        final int width = (int) (1f * bitmaporigi.getWidth() / bitmaporigi.getHeight() * 220);
        final int height = 220;
        final Bitmap bitmaporigiscaled = Bitmap.createScaledBitmap(bitmaporigi, width, height, false);
        final Bitmap bitmappreppcessediscaled = Bitmap.createScaledBitmap(MyBitmap1, width, height, false);
        final int leftOffset = (bitmaporigiscaled.getWidth() - bitmaporigiscaled.getWidth()) / 2;
        final int topOffset = 0;
        /* end resize bitmap*/


        canvas.drawText("Original Image", 20, 200, title);
        canvas.drawBitmap(bitmaporigiscaled, 20, 220, paint);
        //canvas.drawBitmap(bitmaporigiscaled, leftOffset, topOffset, null);
         canvas.drawText("Mean Subtracted Image", 350, 200, title);
        canvas.drawBitmap(MyBitmap1, 350, 220, paint);

        //canvas.drawText("Preprocessed", 20, 740, title);
        //canvas.drawBitmap(MyBitmap1, 20, 760, paint);

       //canvas.drawBitmap(MyBitmap, new Rect(0,0,100,100), rectangle, null);
        // similarly we are creating another text and in this
        // we are aligning this text to center of our PDF file.
        title.setTypeface(Typeface.defaultFromStyle(Typeface.NORMAL));
        title.setColor(ContextCompat.getColor(this, R.color.colorPrimaryDark));
        title.setTextSize(15);

        // below line is used for setting
        // our text to center of PDF.
        title.setTextAlign(Paint.Align.CENTER);
        canvas.drawText("This an autogenerated Ain-DME Report.", 396, 500, title);

        // after adding all attributes to our
        // PDF file we will be finishing our page.
        pdfDocument.finishPage(myPage);

        // below line is used to set the name of
        // our PDF file and its path.
        //File file = new File(Environment.getExternalStorageDirectory(), "GFG.pdf");

       // try {
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        File directory_path2 = cw.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);

        //String directory_path2= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).toString();
            //String directory_path2 = Environment.getExternalStorageDirectory().getPath() + "/mypdf/";

           File file2 = new File(directory_path2.toString());

            if (!file2.exists()) {

                file2.mkdirs();
            }
            //String targetPdf2 = directory_path2;
           String targetPdf2 = directory_path2+"/"+ImagefileName+".pdf";
            File filePath2 = new File(targetPdf2);

            try {

                pdfDocument.writeTo(new FileOutputStream(filePath2));
                Toast.makeText(this, "Report generated successfully.", Toast.LENGTH_LONG).show();

            } catch (IOException e) {
                Log.e("main", "error "+e.toString());
                Toast.makeText(this, "Something wrong: " + e.toString(),  Toast.LENGTH_LONG).show();
            }

            // after creating a file name we will
            // write our PDF file to that location.
           //pdfDocument.writeTo(new FileOutputStream(file));

            // below line is to print toast message
            // on completion of PDF generation.
            //Toast.makeText(MainActivity.this, "PDF file generated successfully.", Toast.LENGTH_SHORT).show();
       // } catch (IOException e) {
            // below line is used
            // to handle error
            //Toast.makeText(MainActivity.this, "PDF  successfully.", Toast.LENGTH_SHORT).show();
           // e.printStackTrace();
       // }
        // after storing our pdf to that
        // location we are closing our PDF file.
        pdfDocument.close();
    }

    private boolean checkPermission() {
        // checking of permissions.
        int permission1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int permission2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        return permission1 == PackageManager.PERMISSION_GRANTED && permission2 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {

        // requesting permissions if not provided.
        ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0) {

                // after requesting permissions we are showing
                // users a toast message of permission granted.
                boolean writeStorage = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                boolean readStorage = grantResults[1] == PackageManager.PERMISSION_GRANTED;

                if (writeStorage && readStorage) {
                    Toast.makeText(this, "Permission Granted..", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "Permission Denied.", Toast.LENGTH_SHORT).show();
                    finish();
                }
            }
        }
    }

///////////////////////////////////////////////////////////////////
}
