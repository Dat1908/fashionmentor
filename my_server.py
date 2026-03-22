# CORS: giup client tu domain khac co the su dung tai nguyen (API) cua Flask, Python
# SS: Flask: Bat SSL cho Backend de dam bao an toan du lieu
# Can co cac file chua khoa va chung chi so SSL
# import re
import os
from random import random
# Import flask
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from html.parser import HTMLParser
from dotenv import load_dotenv

# ── Google GenAI (new SDK) ──
from google import genai as google_genai
from google.genai import types as genai_types

load_dotenv()
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
_genai_client = google_genai.Client(api_key=_GEMINI_KEY) if _GEMINI_KEY else None
_CHAT_MODEL   = "gemini-2.5-flash"
_SEARCH_MODEL = "gemini-2.5-pro"
# Import cac ham chinh
from body_shape_calculator import get_body_shape
from face_shape_detector import load_face_model, get_face_shape
from skin_hair_color_detector import *  
# Face shape classes
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
# Load Model 
model = load_face_model()

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ""

# Giao diện trang chủ
@app.route("/")
def home_page():
    return render_template("home.html")

# Giao diện thông tin của face shape
@app.route("/face_shape/<shape>", methods=['GET', 'POST'])
def face_shape_detail(shape):
    template_name = f"{shape.lower().replace(' ', '_')}.html"
    return render_template(template_name)
    
# Giao diện đoán face shape
@app.route("/face_shape", methods=['GET', 'POST']) # Face Shape
def face_shape_func():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try: 
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file3
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'D:/fashionmentor/image_get' + image.filename)
                # app.config['UPLOAD_FOLDER'] = r"D:/Python/FusionAIVytec2023/static/"  # Dùng 'r' để tránh lỗi escape sequence

                # # Tạo thư mục nếu chưa có
                # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                # # Lọc bỏ ký tự đặc biệt trong tên file
                # safe_filename = re.sub(r'[/*?:"<>|]', '_', image.filename)
                # path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)
                if detect_face(path_to_save) == False:
                    return render_template("face_shape.html", msg="Anh khong hop le")

                face_shape = get_face_shape(model, classes, image_path=path_to_save)
                # skin_color = get_skin_color(path_to_save)
                # hair_color = get_hair_color(path_to_save)
                
                label = f'Face: {face_shape}'
                
                if face_shape in classes:
                    # Trả về kết quả
                    return render_template("face_shape.html", label=label,
                                            msg="Tải file lên thành công")
                else:
                    # Anh chat luong kem
                    return render_template("face_shape.html", 
                                            msg="Vui lòng chọn ảnh khác")
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('face_shape.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('face_shape.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('face_shape.html')


# Giao diện thông tin của body shape
@app.route("/body_shape/<shape>", methods=['GET', 'POST'])
def body_shape_detail(shape):
    template_name = f"{shape.lower().replace(' ', '_')}.html"
    return render_template(template_name)

# Giao diện đoán face shape
@app.route("/body_shape", methods=['GET', 'POST'])
def body_shape_func():
    if request.method == "POST":
        # Lấy thông tin từ các trường input trong form
        bust = request.form.get('Bust')
        waist = request.form.get('Waist')
        hip = request.form.get('Hip')

        # Xử lý dữ liệu đầu vào, ví dụ: tính toán hình dáng cơ thể
        # body_shape = calculate_body_shape(bust, waist, hip)
        
        # Trả về kết quả, bạn có thể chuyển kết quả đó vào template hoặc trả về dạng JSON
        # return render_template('body_shape_result.html', body_shape=body_shape)
        body_shape = get_body_shape(int(bust), int(waist), int(hip))
        return render_template('body_shape.html', body_shape = body_shape, msg = "Thành công!")

    else:
        # Nếu là GET thì hiển thị giao diện form nhập liệu
        return render_template('body_shape.html')

# Giao diện thông tin của personal color
@app.route("/personal_color/<color_name>", methods=['GET', 'POST'])
def personal_color_detail(color_name):
    template_name = f"{color_name.lower().replace(' ', '_')}.html"
    return render_template(template_name)

# Giao diện đoán personal color
@app.route("/personal_color", methods=['GET', 'POST']) # Personal color
def personal_color_func():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'D:/fashionmentor/image_get' + image.filename)
                # # Định nghĩa thư mục lưu file
                # app.config['UPLOAD_FOLDER'] = r"D:/Python/FusionAIVytec2023/static/"  # Dùng 'r' để tránh lỗi escape sequence

                # # Tạo thư mục nếu chưa có
                # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                # # Lọc bỏ ký tự đặc biệt trong tên file
                # # import re
                # safe_filename = re.sub(r'[/*?:"<>|]', '_', image.filename)
                # path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)
                if detect_face(path_to_save) == False:
                    return render_template("personal_color.html", msg="Anh khong hop le")

                skin_color = get_skin_color(path_to_save)
                hair_color = get_hair_color(path_to_save)
                
                # Xu li de ra loai personal color
                # label = str(skin_color) + ' ' + str(hair_color)
                label = personal_color(skin_color, hair_color)
                
                    # Trả về kết quả
                return render_template("personal_color.html", label=label,
                                        msg="Tải file lên thành công")
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('personal_color.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('personal_color.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('personal_color.html')

# ============================================================
# CHATBOT HELPERS
# ============================================================

class _TextExtractor(HTMLParser):
    """Trích xuất text thuần từ HTML (không dùng thư viện ngoài)."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip  = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self):
        return '\n'.join(self._parts)


def _get_page_context(page_name: str) -> str:
    """Đọc template HTML và trả về nội dung text thuần (tối đa 3500 ký tự)."""
    if not page_name:
        return ''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'templates', f'{page_name}.html')
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        parser = _TextExtractor()
        parser.feed(raw)
        return parser.get_text()[:3500]
    except Exception:
        return ''


def _ai_chat(prompt: str) -> str:
    resp = _genai_client.models.generate_content(
        model=_CHAT_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048,
        ),
    )
    return resp.text


def _ai_search(prompt: str) -> str:
    tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
    resp = _genai_client.models.generate_content(
        model=_SEARCH_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            tools=[tool],
            temperature=0.3,
        ),
    )
    return resp.text


# ============================================================
# CHATBOT ROUTES
# ============================================================

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/api/chat', methods=['POST'])
def chatbot_api():
    if not _genai_client:
        return jsonify({'error': 'Thiếu GEMINI_API_KEY trong file .env'}), 500

    data   = request.get_json(force=True) or {}
    prompt = data.get('prompt', '').strip()
    mode   = data.get('mode', 'chat')   # 'chat' | 'search'
    page   = data.get('page', '').strip()

    if not prompt:
        return jsonify({'error': 'Prompt không được để trống'}), 400

    context = _get_page_context(page)

    if context:
        full_prompt = (
            f"Bạn là trợ lý thời trang AI thông minh của FashionMentor. "
            f"Dựa trên nội dung thời trang sau đây:\n\n"
            f"---\n{context}\n---\n\n"
            f"Hãy trả lời câu hỏi bằng tiếng Việt một cách chi tiết và hữu ích.\n"
            f"Câu hỏi: {prompt}"
        )
    else:
        full_prompt = (
            f"Bạn là trợ lý thời trang AI của FashionMentor. "
            f"Hãy trả lời bằng tiếng Việt: {prompt}"
        )

    try:
        if mode == 'search':
            result = _ai_search(full_prompt)
        else:
            result = _ai_chat(full_prompt)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# AUTO-INJECT Chatbot Widget vào mọi trang HTML kết quả
# ============================================================

WIDGET_SCRIPT = '\n<script src="/static/js/chatbot_widget.js" defer></script>\n'
EXCLUDED_PATHS = {'/chatbot', '/api/chat'}

@app.after_request
def inject_chatbot_widget(response):
    """Tự động chèn floating chatbot widget vào cuối mọi trang HTML."""
    if request.path in EXCLUDED_PATHS:
        return response
    if 'text/html' not in response.content_type:
        return response
    content = response.get_data(as_text=True)
    if '</body>' in content:
        content = content.replace('</body>', WIDGET_SCRIPT + '</body>', 1)
        response.set_data(content)
    return response


# ============================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)