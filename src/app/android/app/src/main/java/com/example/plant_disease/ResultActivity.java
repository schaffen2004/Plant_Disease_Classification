package com.example.plant_disease;

import androidx.appcompat.app.AppCompatActivity;

import android.text.Html;
import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ResultActivity extends AppCompatActivity {

    private static final String API_URL = "https://resulted-urgent-mortality-mentor.trycloudflare.com/predict";

    private ImageView resultImage;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        resultImage = findViewById(R.id.resultImage);
        resultText = findViewById(R.id.resultText);

        String uriString = getIntent().getStringExtra("imageUri");
        if (uriString != null) {
            Uri imageUri = Uri.parse(uriString);
            resultImage.setImageURI(imageUri);

            try {
                File imageFile = prepareImageFile(imageUri);
                if (imageFile != null && imageFile.exists()) {
                    sendImageToApi(imageFile);
                } else {
                    resultText.setText("❌ Không thể xử lý ảnh!");
                }
            } catch (IOException e) {
                resultText.setText("❌ Lỗi đọc ảnh: " + e.getMessage());
                e.printStackTrace();
            }
        } else {
            resultText.setText("Không tìm thấy ảnh!");
        }
    }

    /**
     * Chuẩn hóa ảnh: resize về 256x256 và lưu dưới dạng JPEG
     */
    private File prepareImageFile(Uri uri) throws IOException {
        ContentResolver resolver = getContentResolver();
        InputStream inputStream = resolver.openInputStream(uri);
        if (inputStream == null) return null;

        Bitmap originalBitmap = BitmapFactory.decodeStream(inputStream);
        inputStream.close();
        if (originalBitmap == null) return null;

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 256, 256, true);
        File resizedFile = new File(getCacheDir(), "plant_" + System.currentTimeMillis() + ".jpg");
        FileOutputStream out = new FileOutputStream(resizedFile);
        resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out);
        out.flush();
        out.close();
        return resizedFile;
    }

    /**
     * Gửi ảnh đến API
     */
    private void sendImageToApi(File imageFile) {
        if (imageFile == null || !imageFile.exists()) {
            resultText.setText("❌ File ảnh không hợp lệ!");
            return;
        }

        OkHttpClient client = new OkHttpClient();

        RequestBody fileBody = RequestBody.create(
                imageFile,
                MediaType.parse("image/jpeg")
        );

        MultipartBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFile.getName(), fileBody)
                .build();

        Request request = new Request.Builder()
                .url(API_URL)
                .post(requestBody)
                .build();

        resultText.setText("🌱 Đang xử lý ảnh...");

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                runOnUiThread(() -> {
                    resultText.setText("❌ Lỗi gửi request: " + e.getMessage());
                    Log.e("API_ERROR", e.toString());
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    runOnUiThread(() -> resultText.setText("❌ Server lỗi: " + response.code()));
                    return;
                }

                String resString = response.body().string();
                Log.i("API_RESPONSE", resString);

                try {
                    JSONObject json = new JSONObject(resString);

                    // ✅ Đọc đúng key mới từ API
                    String predictedClass = json.optString("predicted_class", "Unknown");
                    double confidence = json.optDouble("confidence", 0.0);
                    StringBuilder displayText = new StringBuilder();
                    if(confidence<0.80){

                        displayText.append("🌿 <b>Không chắc chắn về kết quả!</b><br>");
                        displayText.append("Ảnh chưa rõ ràng hoặc lá không phải loại trong dữ liệu.<br><br>");
                        displayText.append("👉 Hãy thử lại bằng cách:<br>");
                        displayText.append("- Chụp rõ 1 lá duy nhất<br>");
                        displayText.append("- Đảm bảo ánh sáng tốt, không bị mờ<br>");
                    }
                    else{
                        displayText.append("🌿 <b>Kết quả dự đoán:</b><br>");
                        displayText.append("<b>Bệnh:</b> ").append(predictedClass).append("<br>");
                        displayText.append("<b>Độ tin cậy:</b> ")
                                .append(String.format("%.2f", confidence * 100)).append("%<br><br>");
                    }

// Dùng HTML để in đẹp hơn
                    runOnUiThread(() ->
                            resultText.setText(Html.fromHtml(displayText.toString(), Html.FROM_HTML_MODE_LEGACY))
                    );

                } catch (JSONException ex) {
                    runOnUiThread(() -> resultText.setText("⚠️ JSON không đúng định dạng: " + ex.getMessage()));
                }
            }
        });
    }
}