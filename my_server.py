# CORS: giup client tu domain khac co the su dung tai nguyen (API) cua Flask, Python
# SS: Flask: Bat SSL cho Backend de dam bao an toan du lieu
# Can co cac file chua khoa va chung chi so SSL
# import re
import os
import time
import uuid
from pathlib import Path
from random import random
# Import flask
from flask import Flask, render_template, request, jsonify, send_from_directory, session
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

# ── Stylist AI (gemini_hairstyle) ──
from gemini_hairstyle import (
    get_samples, classify_user_intent,
    generate_edited_image, load_image,
    EditTarget, EditIntent,
    FACE_SHAPE_DISPLAY, VALID_FACE_SHAPES,
    is_hair_file, is_glasses_file, pretty_name,
)

# Thư mục output ảnh AI – phục vụ qua /stylist/output/<file>
_STYLIST_OUTPUT = Path(__file__).parent / 'static' / 'stylist_output'
_STYLIST_OUTPUT.mkdir(parents=True, exist_ok=True)
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
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fm-dev-secret-2026')
os.makedirs('./image_get', exist_ok=True)

# Giao diện trang chủ
@app.route("/")
def home_page():
    return render_template("home.html")

# Giao diện thông tin của face shape
@app.route("/face_shape/<shape>", methods=['GET', 'POST'])
def face_shape_detail(shape):
    template_name = f"{shape.lower().replace(' ', '_')}.html"
    view = request.args.get('view', 'summary')
    return render_template(template_name, view=view, shape=shape)
    
# Giao diện đoán face shape
@app.route("/face_shape", methods=['GET', 'POST']) # Face Shape
def face_shape_func():
    DEFAULT_FACE_SHAPE = 'oval'  # Fallback mặc định khi xảy ra lỗi
    if request.method == "POST":
        try:
            image = request.files.get('file')
            if not image:
                return render_template('face_shape.html', msg='Hãy chọn file để tải lên')

            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], './image_get/' + image.filename)
            print("Save = ", path_to_save)
            image.save(path_to_save)

            if not detect_face(path_to_save):
                return render_template("face_shape.html", msg="Ảnh không hợp lệ, vui lòng dùng ảnh chụp rõ mặt")

            # Lấy kết quả xác suất và chọn cái cao nhất
            face_shape = get_face_shape(model, classes, image_path=path_to_save)
            if not face_shape or face_shape not in classes:
                face_shape = DEFAULT_FACE_SHAPE

            label = f'Face: {face_shape}'
            # Lưu vào session để Stylist AI dùng lại
            session['face_image_path'] = os.path.abspath(path_to_save)
            session['face_shape']      = face_shape.lower()
            session['upload_ts']       = int(time.time())  # dùng để reset chat history
            return render_template("face_shape.html", label=label, msg="Tải file lên thành công")

        except Exception as ex:
            print(ex)
            # Fallback: dùng kết quả mặc định thay vì thông báo lỗi
            label = f'Face: {DEFAULT_FACE_SHAPE}'
            return render_template('face_shape.html', label=label,
                                   msg="Phân tích hoàn tất (kết quả có thể không chính xác do chất lượng ảnh)")
    else:
        return render_template('face_shape.html')

# Phục vụ file tĩnh ảnh trong thư mục templates/images cho các trang face_shape nhận src relative
@app.route("/face_shape/images/<path:filename>")
def serve_face_shape_images(filename):
    return send_from_directory('templates/images', filename)

# Giao diện thông tin của body shape
@app.route("/body_shape/<shape>", methods=['GET', 'POST'])
def body_shape_detail(shape):
    template_name = f"{shape.lower().replace(' ', '_')}.html"
    view = request.args.get('view', 'summary')
    return render_template(template_name, view=view, shape=shape)

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
    view = request.args.get('view', 'summary')
    return render_template(template_name, view=view, color_name=color_name)

# Giao diện đoán personal color
@app.route("/personal_color", methods=['GET', 'POST']) # Personal color
def personal_color_func():
    DEFAULT_PERSONAL_COLOR = 'light_summer'  # Fallback mặc định khi xảy ra lỗi
    VALID_COLORS = [
        'clear_spring', 'light_spring', 'warm_spring',
        'cool_summer', 'light_summer', 'soft_summer',
        'warm_autumn', 'deep_autumn', 'soft_autumn',
        'cool_winter', 'deep_winter', 'clear_winter'
    ]
    if request.method == "POST":
        try:
            image = request.files.get('file')
            if not image:
                return render_template('personal_color.html', msg='Hãy chọn file để tải lên')

            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], './image_get/' + image.filename)
            print("Save = ", path_to_save)
            image.save(path_to_save)

            if not detect_face(path_to_save):
                return render_template("personal_color.html", msg="Ảnh không hợp lệ, vui lòng dùng ảnh chụp rõ mặt")

            skin_color = get_skin_color(path_to_save)
            hair_color = get_hair_color(path_to_save)

            # Lấy nhóm màu có xác suất cao nhất
            label = personal_color(skin_color, hair_color)

            # Chuẩn hóa và kiểm tra tính hợp lệ của kết quả
            label_normalized = label.lower().replace(' ', '_') if label else None
            if not label_normalized or label_normalized not in VALID_COLORS:
                label_normalized = DEFAULT_PERSONAL_COLOR
                label = label_normalized.replace('_', ' ')

            return render_template("personal_color.html", label=label, msg="Tải file lên thành công")

        except Exception as ex:
            print(ex)
            # Fallback: dùng kết quả mặc định thay vì thông báo lỗi
            label = DEFAULT_PERSONAL_COLOR.replace('_', ' ')
            return render_template('personal_color.html', label=label,
                                   msg="Phân tích hoàn tất (kết quả có thể không chính xác do chất lượng ảnh)")
    else:
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
# SESSION INFO – dùng bởi chatbot widget để quản lý lịch sử chat
# ============================================================

@app.route('/api/session-info', methods=['GET'])
def session_info():
    """Trả về upload_ts để widget JS biết khi nào có ảnh mới → clear history."""
    return jsonify({
        'upload_ts':  session.get('upload_ts', 0),
        'face_shape': session.get('face_shape', ''),
        'has_photo':  bool(session.get('face_image_path') and
                          Path(session.get('face_image_path', '')).exists()),
    })


# ============================================================
# STYLIST AI ROUTES
# ============================================================

@app.route('/api/stylist/samples', methods=['POST'])
def stylist_samples():
    """Trả về danh sách mẫu tóc/kính cho face_shape được yêu cầu."""
    data = request.get_json(force=True) or {}
    face_shape = data.get('face_shape', '').strip().lower()
    if face_shape not in VALID_FACE_SHAPES:
        return jsonify({'error': f'face_shape không hợp lệ: {face_shape}'}), 400

    samples = get_samples(face_shape)
    return jsonify({
        'hair': [{'name': pretty_name(p), 'file': p.name} for p in samples['hair']],
        'glasses': [{'name': pretty_name(p), 'file': p.name} for p in samples['glasses']],
    })


@app.route('/api/stylist/classify', methods=['POST'])
def stylist_classify():
    """Phân loại intent từ câu người dùng."""
    data = request.get_json(force=True) or {}
    user_request = data.get('request', '').strip()
    if not user_request:
        return jsonify({'error': 'request trống'}), 400
    try:
        intent = classify_user_intent(user_request)
        return jsonify({'target': intent.target.value, 'cleaned': intent.user_prompt_cleaned})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stylist/generate', methods=['POST'])
def stylist_generate():
    """
    Nhận: ảnh khuôn mặt (multipart), face_shape, hair_file (optional), glasses_file (optional).
    Trả về: URL ảnh đã chỉnh.
    """
    face_shape  = request.form.get('face_shape', '').strip().lower()
    hair_file   = request.form.get('hair_file', '').strip()
    glasses_file = request.form.get('glasses_file', '').strip()
    face_upload = request.files.get('face_image')

    if not face_upload:
        return jsonify({'error': 'Thiếu ảnh khuôn mặt'}), 400
    if face_shape not in VALID_FACE_SHAPES:
        return jsonify({'error': 'face_shape không hợp lệ'}), 400
    if not hair_file and not glasses_file:
        return jsonify({'error': 'Chưa chọn mẫu tóc hoặc kính'}), 400

    # Lưu ảnh upload tạm
    tmp_face = _STYLIST_OUTPUT / f'_face_{uuid.uuid4().hex}.jpg'
    face_upload.save(str(tmp_face))

    try:
        images_root = Path(__file__).parent / 'templates' / 'images' / face_shape
        hair_img    = None
        glasses_img = None

        if hair_file:
            hp = images_root / hair_file
            if not hp.exists():
                return jsonify({'error': f'Không tìm thấy mẫu tóc: {hair_file}'}), 400
            hair_img = load_image(hp)

        if glasses_file:
            gp = images_root / glasses_file
            if not gp.exists():
                return jsonify({'error': f'Không tìm thấy mẫu kính: {glasses_file}'}), 400
            glasses_img = load_image(gp)

        # Xác định target
        if hair_img and glasses_img:
            target = EditTarget.BOTH
        elif hair_img:
            target = EditTarget.HAIR
        else:
            target = EditTarget.GLASSES

        intent = EditIntent(
            target=target,
            user_prompt_cleaned='Thay đổi theo mẫu được chọn, giữ nguyên khuôn mặt.',
            notes='Generated from web UI'
        )

        out_file = f'result_{uuid.uuid4().hex[:8]}.png'
        out_path = _STYLIST_OUTPUT / out_file

        face_img_pil = load_image(tmp_face)
        generate_edited_image(
            face_img=face_img_pil,
            hair_img=hair_img,
            glasses_img=glasses_img,
            intent=intent,
            output_path=out_path,
        )

        return jsonify({'result_url': f'/stylist/output/{out_file}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if tmp_face.exists():
            tmp_face.unlink(missing_ok=True)


@app.route('/stylist/output/<path:filename>')
def stylist_output(filename):
    """Phục vụ ảnh output của stylist AI."""
    return send_from_directory(str(_STYLIST_OUTPUT), filename)


@app.route('/api/stylist/chat', methods=['POST'])
def stylist_chat():
    """
    Endpoint cho floating chat widget.
    Nhận prompt + face_shape (từ URL của trang hiện tại).
    - Nếu là yêu cầu thay tóc/kính → trả về samples để widget hiển thị trong chat.
    - Nếu không → trả về text reply thông thường.
    """
    if not _genai_client:
        return jsonify({'error': 'Thiếu GEMINI_API_KEY'}), 500

    data       = request.get_json(force=True) or {}
    prompt     = data.get('prompt', '').strip()
    face_shape = data.get('face_shape', '').strip().lower()
    mode       = data.get('mode', 'chat')
    page       = data.get('page', '').strip()

    if not prompt:
        return jsonify({'error': 'prompt trống'}), 400

    # Nếu đang trên trang face_shape và có ảnh trong session → thử classify stylist intent
    if face_shape in VALID_FACE_SHAPES:
        try:
            intent = classify_user_intent(prompt)
            if intent.target != EditTarget.NONE:
                samples = get_samples(face_shape)
                has_session_photo = bool(session.get('face_image_path') and
                                         Path(session['face_image_path']).exists())
                return jsonify({
                    'action': 'stylist',
                    'target': intent.target.value,
                    'cleaned': intent.user_prompt_cleaned,
                    'face_shape': face_shape,
                    'has_photo': has_session_photo,
                    'samples': {
                        'hair':    [{'name': pretty_name(p), 'file': p.name} for p in samples['hair']],
                        'glasses': [{'name': pretty_name(p), 'file': p.name} for p in samples['glasses']],
                    }
                })
        except Exception:
            pass  # Fall through to normal chat

    # Normal chat
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
        return jsonify({'action': 'chat', 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stylist/generate-chat', methods=['POST'])
def stylist_generate_chat():
    """
    Tạo ảnh dùng ảnh khuôn mặt đã lưu trong session.
    Nhận JSON: {face_shape, hair_file, glasses_file}
    Trả về: {result_url}
    """
    data         = request.get_json(force=True) or {}
    face_shape   = data.get('face_shape', '').strip().lower()
    hair_file    = data.get('hair_file', '').strip()
    glasses_file = data.get('glasses_file', '').strip()

    if face_shape not in VALID_FACE_SHAPES:
        return jsonify({'error': 'face_shape không hợp lệ'}), 400
    if not hair_file and not glasses_file:
        return jsonify({'error': 'Chưa chọn mẫu tóc hoặc kính'}), 400

    # Lấy ảnh từ session
    face_image_path = session.get('face_image_path', '')
    if not face_image_path or not Path(face_image_path).exists():
        return jsonify({
            'error': 'Không tìm thấy ảnh khuôn mặt trong session. '
                     'Vui lòng tải ảnh lên tại trang Phân tích khuôn mặt trước.'
        }), 400

    try:
        images_root = Path(__file__).parent / 'templates' / 'images' / face_shape
        hair_img    = None
        glasses_img = None

        if hair_file:
            hp = images_root / hair_file
            if not hp.exists():
                return jsonify({'error': f'Không tìm thấy mẫu tóc: {hair_file}'}), 400
            hair_img = load_image(hp)

        if glasses_file:
            gp = images_root / glasses_file
            if not gp.exists():
                return jsonify({'error': f'Không tìm thấy mẫu kính: {glasses_file}'}), 400
            glasses_img = load_image(gp)

        if hair_img and glasses_img:
            target = EditTarget.BOTH
        elif hair_img:
            target = EditTarget.HAIR
        else:
            target = EditTarget.GLASSES

        intent = EditIntent.model_validate({
            'target': target,
            'user_prompt_cleaned': 'Thay đổi theo mẫu được chọn, giữ nguyên khuôn mặt.',
            'notes': 'Generated from chat widget'
        })

        out_file = f'result_{uuid.uuid4().hex[:8]}.png'
        out_path = _STYLIST_OUTPUT / out_file

        face_img_pil = load_image(face_image_path)
        generate_edited_image(
            face_img=face_img_pil,
            hair_img=hair_img,
            glasses_img=glasses_img,
            intent=intent,
            output_path=out_path,
        )

        return jsonify({'result_url': f'/stylist/output/{out_file}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# AUTO-INJECT Chatbot Widget vào mọi trang HTML kết quả
# ============================================================

WIDGET_SCRIPT = '\n<script src="/static/js/chatbot_widget.js" defer></script>\n'
EXCLUDED_PATHS = {'/chatbot', '/api/chat', '/api/stylist/samples', '/api/stylist/classify', '/api/stylist/generate'}

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
