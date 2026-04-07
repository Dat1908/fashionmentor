/**
 * FashionMentor AI – Floating Chatbot Widget
 * Supports: normal chat/search + in-chat Stylist AI (hair/glasses swap)
 */
(function () {
    'use strict';

    /* ── 1. CSS ── */
    const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#fm-fab {
    position: fixed; bottom: 28px; right: 28px; z-index: 9999;
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.55);
    transition: transform 0.25s, box-shadow 0.25s;
    font-size: 22px; color: #fff; user-select: none;
}
#fm-fab:hover { transform: scale(1.1); box-shadow: 0 6px 28px rgba(124,58,237,0.7); }
#fm-fab .fm-badge {
    position: absolute; top: -4px; right: -4px;
    background: #ec4899; color: #fff; font-size: 10px; font-weight: 700;
    width: 18px; height: 18px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    border: 2px solid #07070f; opacity: 0; transition: opacity 0.2s;
}
#fm-fab .fm-badge.show { opacity: 1; }

#fm-panel {
    position: fixed; bottom: 96px; right: 28px; z-index: 9998;
    width: 390px; height: 560px; border-radius: 20px;
    background: rgba(10,10,20,0.97); backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(124,58,237,0.2);
    display: flex; flex-direction: column;
    font-family: 'Inter', sans-serif; overflow: hidden;
    transform: scale(0.85) translateY(20px);
    transform-origin: bottom right; opacity: 0; pointer-events: none;
    transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1), opacity 0.22s ease;
}
#fm-panel.open { transform: scale(1) translateY(0); opacity: 1; pointer-events: all; }

/* Header */
.fm-header {
    padding: 14px 16px 10px; border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0; background: rgba(255,255,255,0.025);
}
.fm-header-left { display: flex; align-items: center; gap: 10px; }
.fm-avatar-hd {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.fm-header-info .fm-title { font-size: 13.5px; font-weight: 600; color: #f1f5f9; }
.fm-header-info .fm-sub   { font-size: 11px; color: #6b7280; margin-top: 1px; }
.fm-header-actions { display: flex; align-items: center; gap: 6px; }

/* Mode toggle */
.fm-mode-toggle {
    display: flex; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 30px;
    padding: 2px; gap: 1px;
}
.fm-mode-btn {
    padding: 4px 11px; border-radius: 30px; border: none;
    font-size: 11px; font-weight: 500; cursor: pointer;
    transition: all 0.2s; background: transparent; color: #9ca3af;
    font-family: 'Inter', sans-serif; white-space: nowrap;
}
.fm-mode-btn.active {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: #fff; box-shadow: 0 1px 10px rgba(124,58,237,0.4);
}
.fm-close-btn {
    width: 28px; height: 28px; border-radius: 50%; border: none;
    background: rgba(255,255,255,0.07); color: #9ca3af; font-size: 14px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background 0.2s; font-family: 'Inter', sans-serif;
}
.fm-close-btn:hover { background: rgba(255,255,255,0.14); color: #f1f5f9; }

/* Messages */
.fm-messages {
    flex: 1; overflow-y: auto;
    padding: 14px 14px 6px;
    display: flex; flex-direction: column; gap: 12px;
}
.fm-messages::-webkit-scrollbar { width: 3px; }
.fm-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* Welcome */
.fm-welcome {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; gap: 10px; opacity: 0.7; padding: 20px;
}
.fm-welcome .fw-icon { font-size: 32px; }
.fm-welcome p { font-size: 12.5px; color: #6b7280; line-height: 1.55; }

/* Message bubbles */
.fm-msg { display: flex; gap: 8px; animation: fmSlideUp 0.25s ease; }
.fm-msg.user { flex-direction: row-reverse; }
@keyframes fmSlideUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}
.fm-av {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 13px;
}
.fm-av.ai   { background: linear-gradient(135deg,#7c3aed,#db2777); }
.fm-av.user { background: rgba(255,255,255,0.1); }
.fm-bubble {
    max-width: 84%; padding: 9px 12px;
    border-radius: 14px; font-size: 12.5px; line-height: 1.6;
}
.fm-bubble.ai {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.08);
    border-top-left-radius: 4px; color: #e2e8f0;
}
.fm-bubble.ai h1,.fm-bubble.ai h2,.fm-bubble.ai h3 { color:#c4b5fd; font-size:13px; margin:8px 0 4px; }
.fm-bubble.ai ul,.fm-bubble.ai ol { padding-left:16px; margin:4px 0; }
.fm-bubble.ai li { margin:2px 0; }
.fm-bubble.ai a  { color:#a78bfa; text-decoration:underline; }
.fm-bubble.ai strong { color:#f0abfc; }
.fm-bubble.ai code { background:rgba(0,0,0,0.3); padding:1px 5px; border-radius:4px; font-size:11px; }
.fm-bubble.user {
    background: linear-gradient(135deg,#7c3aed,#db2777);
    color:#fff; border-top-right-radius:4px;
}
.fm-badge-mode {
    font-size:10px; padding:1px 7px; border-radius:20px;
    margin-bottom:5px; display:inline-block;
    background:rgba(255,255,255,0.07); color:#6b7280;
}
.fm-badge-mode.search { background:rgba(16,185,129,0.12); color:#34d399; }
.fm-badge-mode.stylist { background:rgba(168,85,247,0.15); color:#c4b5fd; }

/* Loading dots */
.fm-dots { display:flex; gap:4px; padding:2px 0; align-items:center; }
.fm-dots span { width:7px; height:7px; border-radius:50%; animation:fmBounce 1.3s infinite; }
.fm-dots span:nth-child(1){background:#7c3aed;}
.fm-dots span:nth-child(2){background:#a855f7;animation-delay:.2s;}
.fm-dots span:nth-child(3){background:#db2777;animation-delay:.4s;}
@keyframes fmBounce {
    0%,60%,100%{transform:translateY(0);opacity:.5;}
    30%{transform:translateY(-7px);opacity:1;}
}

/* ── Stylist picker inside bubble ── */
.fm-stylist-section { margin-top: 10px; }
.fm-stylist-label {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: #6b7280; margin-bottom: 6px;
}
.fm-sample-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(78px, 1fr));
    gap: 6px;
}
.fm-sample-card {
    border-radius: 9px; overflow: hidden;
    border: 2px solid rgba(255,255,255,0.08);
    cursor: pointer; transition: all 0.16s;
    background: rgba(255,255,255,0.04);
    position: relative;
}
.fm-sample-card:hover { border-color: rgba(168,85,247,0.5); transform: translateY(-1px); }
.fm-sample-card.selected {
    border-color: #a855f7;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.25);
}
.fm-sample-card img { width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }
.fm-sample-card .fsc-name {
    font-size: 9px; color: #9ca3af;
    padding: 3px 4px; text-align: center; line-height: 1.3;
}
.fm-sample-card .fsc-check {
    position: absolute; top: 3px; right: 3px;
    background: #a855f7; color: #fff; border-radius: 50%;
    width: 14px; height: 14px; font-size: 9px;
    display: none; align-items: center; justify-content: center;
}
.fm-sample-card.selected .fsc-check { display: flex; }

.fm-gen-btn {
    width: 100%; margin-top: 10px;
    padding: 10px; border-radius: 10px; border: none;
    background: linear-gradient(135deg,#7c3aed,#db2777);
    color: #fff; font-size: 12.5px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 3px 14px rgba(124,58,237,0.4);
}
.fm-gen-btn:hover:not(:disabled) { transform:translateY(-1px); box-shadow:0 5px 18px rgba(124,58,237,0.55); }
.fm-gen-btn:disabled { opacity:0.4; cursor:not-allowed; }

.fm-no-photo-warn {
    font-size: 11px; color: #f87171;
    background: rgba(239,68,68,0.1); border-radius: 8px;
    padding: 7px 10px; margin-top: 8px; line-height: 1.5;
}
.fm-result-img {
    width: 100%; border-radius: 10px; margin-top: 10px;
    border: 1px solid rgba(168,85,247,0.35);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.fm-result-actions {
    display: flex; gap: 6px; margin-top: 6px;
}
.fm-result-actions a {
    flex: 1; padding: 6px; border-radius: 8px; text-align: center;
    font-size: 11px; font-weight: 500; text-decoration: none;
    background: rgba(168,85,247,0.15);
    color: #c4b5fd; border: 1px solid rgba(168,85,247,0.25);
    transition: background 0.18s;
}
.fm-result-actions a:hover { background: rgba(168,85,247,0.28); }

.fm-empty { font-size: 11px; color: #4b5563; font-style: italic; padding: 4px 0; }

/* Input area */
.fm-input-area {
    padding: 10px 12px 12px; border-top: 1px solid rgba(255,255,255,0.07);
    flex-shrink: 0; background: rgba(255,255,255,0.015);
}
.fm-input-box {
    display: flex; gap: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 7px 8px 7px 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.fm-input-box:focus-within {
    border-color: rgba(168,85,247,0.5);
    box-shadow: 0 0 0 3px rgba(168,85,247,0.1);
}
.fm-input-box textarea {
    flex: 1; background: none; border: none; outline: none;
    font-family: 'Inter',sans-serif; font-size: 12.5px;
    color: #e2e8f0; resize: none; max-height: 90px; min-height: 20px; line-height: 1.5;
}
.fm-input-box textarea::placeholder { color: #374151; }
.fm-send {
    width: 32px; height: 32px; border-radius: 9px; border: none;
    background: linear-gradient(135deg,#7c3aed,#db2777);
    color: #fff; cursor: pointer; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s; font-size: 13px;
}
.fm-send:hover:not(:disabled) { transform:scale(1.08); box-shadow:0 3px 14px rgba(124,58,237,0.5); }
.fm-send:disabled { opacity:0.35; cursor:not-allowed; }
.fm-hint { font-size:10px; color:#374151; text-align:center; margin-top:6px; }
`;

    const styleEl = document.createElement('style');
    styleEl.textContent = CSS;
    document.head.appendChild(styleEl);

    /* ── 2. Detect page context ── */
    const path    = window.location.pathname;
    let pageName  = '';
    let topicName = 'thời trang';
    let faceShape = '';   // e.g. 'round'

    const m = path.match(/\/(face_shape|body_shape|personal_color)\/(.+)/);
    if (m) {
        pageName  = m[2].replace(/-/g, '_');
        topicName = pageName.replace(/_/g, ' ');
        if (m[1] === 'face_shape') {
            faceShape = m[2].toLowerCase();  // 'round', 'oval', etc.
        }
    }

    const VALID_SHAPES = new Set(['heart','oblong','oval','round','square']);
    const isStylistPage = VALID_SHAPES.has(faceShape);

    /* ── 3. Build DOM ── */
    const fab = document.createElement('button');
    fab.id = 'fm-fab';
    fab.title = 'Mở FashionMentor AI';
    fab.innerHTML = '💬<span class="fm-badge" id="fm-badge">1</span>';
    document.body.appendChild(fab);

    const panel = document.createElement('div');
    panel.id = 'fm-panel';
    panel.innerHTML = `
<div class="fm-header">
    <div class="fm-header-left">
        <div class="fm-avatar-hd">✨</div>
        <div class="fm-header-info">
            <div class="fm-title">FashionMentor AI</div>
            <div class="fm-sub" id="fm-sub">Đang xem: ${topicName || 'trang chính'}</div>
        </div>
    </div>
    <div class="fm-header-actions">
        <div class="fm-mode-toggle">
            <button class="fm-mode-btn active" id="fm-btn-chat" onclick="fmSetMode('chat')">💬 Chat</button>
            <button class="fm-mode-btn" id="fm-btn-search" onclick="fmSetMode('search')">🔍 Search</button>
        </div>
        <button class="fm-close-btn" id="fm-close" title="Đóng">✕</button>
    </div>
</div>
<div class="fm-messages" id="fm-messages">
    <div class="fm-welcome" id="fm-welcome">
        <div class="fw-icon">✨</div>
        <p>${isStylistPage
            ? `Tôi là trợ lý AI của FashionMentor.<br>
               Bạn có thể hỏi tôi về thời trang, hoặc nói <b style="color:#c4b5fd">
               "đổi tóc"</b>, <b style="color:#c4b5fd">"thử kính"</b> để tôi gợi ý kiểu
               phù hợp với khuôn mặt <b style="color:#c4b5fd">${topicName}</b> của bạn (dùng ảnh đã quét)!`
            : `Xin chào! Tôi là trợ lý AI thời trang.<br>
               ${pageName ? `Tôi sẽ tư vấn dựa trên nội dung trang <b style="color:#c4b5fd">${topicName}</b> này.` : 'Hãy hỏi tôi về thời trang!'}`
        }</p>
    </div>
</div>
<div class="fm-input-area">
    <div class="fm-input-box">
        <textarea id="fm-input" placeholder="${isStylistPage ? 'Hỏi về thời trang hoặc "đổi tóc", "thử kính"...' : 'Hỏi về thời trang...'}" rows="1"></textarea>
        <button class="fm-send" id="fm-send" title="Gửi">➤</button>
    </div>
    <div class="fm-hint" id="fm-hint">💬 Chat · Gemini 2.5 Flash</div>
</div>`;
    document.body.appendChild(panel);

    /* ── 4. State ── */
    let isOpen    = false;
    let mode      = 'chat';
    let busy      = false;
    let hasUnread = false;

    // Stylist per-session state
    const stylistState = {};   // { bubbleId: { faceShape, selectedHair, selectedGlasses } }
    let _bubbleCounter = 0;

    /* ── 4b. Chat history (localStorage) ── */
    const HISTORY_KEY = 'fm_chat_' + (pageName || 'general');
    const TS_KEY      = 'fm_upload_ts';
    let   _history    = [];  // [{role, html, badgeType, isText}]

    function saveHistory() {
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(_history.slice(-60))); } catch(e) {}
    }

    function restoreHistory() {
        try {
            const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
            if (!saved.length) return;
            removeWelcome();
            saved.forEach(rec => {
                _addMsgDom(rec.role, rec.html, rec.badgeType);
            });
            _history = saved;
            const msgs = document.getElementById('fm-messages');
            if (msgs) msgs.scrollTop = msgs.scrollHeight;
        } catch(e) {}
    }

    async function initHistory() {
        try {
            const r    = await fetch('/api/session-info');
            const info = await r.json();
            const storedTs = localStorage.getItem(TS_KEY);
            const serverTs = String(info.upload_ts || 0);
            if (serverTs !== '0' && storedTs !== serverTs) {
                // New upload detected – clear old history for this page
                localStorage.removeItem(HISTORY_KEY);
                localStorage.setItem(TS_KEY, serverTs);
            } else {
                restoreHistory();
            }
        } catch(e) {
            restoreHistory();
        }
    }

    /* ── 5. marked.js ── */
    let markedReady = false;
    function loadMarked(cb) {
        if (markedReady) { cb(); return; }
        const s = document.createElement('script');
        s.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
        s.onload = () => { markedReady = true; cb(); };
        document.head.appendChild(s);
    }
    function parseMd(text) {
        if (typeof marked !== 'undefined') return marked.parse(text);
        return text.replace(/\n/g,'<br>');
    }

    /* ── 6. Open / Close ── */
    function togglePanel() {
        isOpen = !isOpen;
        panel.classList.toggle('open', isOpen);
        fab.innerHTML = isOpen
            ? '✕<span class="fm-badge" id="fm-badge"></span>'
            : '💬<span class="fm-badge" id="fm-badge"' + (hasUnread ? ' class="show"' : '') + '></span>';
        if (isOpen) {
            hasUnread = false;
            document.getElementById('fm-input').focus();
        }
    }

    fab.addEventListener('click', togglePanel);
    document.getElementById('fm-close').addEventListener('click', togglePanel);

    /* ── 7. Mode ── */
    window.fmSetMode = function (m2) {
        mode = m2;
        document.getElementById('fm-btn-chat').classList.toggle('active', m2 === 'chat');
        document.getElementById('fm-btn-search').classList.toggle('active', m2 === 'search');
        document.getElementById('fm-hint').textContent =
            m2 === 'chat' ? '💬 Chat · Gemini 2.5 Flash' : '🔍 Web Search · Gemini 2.5 Pro';
    };

    /* ── 8. Helpers ── */
    function escHtml(t) {
        return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    }

    function removeWelcome() {
        const el = document.getElementById('fm-welcome');
        if (el) el.remove();
    }

    function _addMsgDom(role, html, badgeType) {
        const msgs = document.getElementById('fm-messages');
        const div  = document.createElement('div');
        div.className = `fm-msg ${role}`;
        if (role === 'ai') {
            const bCls = badgeType === 'search' ? 'search' : badgeType === 'stylist' ? 'stylist' : '';
            const bTxt = badgeType === 'search' ? '🔍 Search' : badgeType === 'stylist' ? '✨ Stylist AI' : '💬 Chat';
            div.innerHTML = `
                <div class="fm-av ai">✨</div>
                <div class="fm-bubble ai">
                    <div class="fm-badge-mode ${bCls}">${bTxt}</div>
                    ${html}
                </div>`;
        } else {
            div.innerHTML = `
                <div class="fm-av user">👤</div>
                <div class="fm-bubble user">${escHtml(html)}</div>`;
        }
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
        return div;
    }

    function addMsg(role, html, badgeType) {
        removeWelcome();
        const div = _addMsgDom(role, html, badgeType);
        // Lưu vào history (chỉ lưu text messages, không lưu stylist interactive bubbles)
        if (badgeType !== 'stylist') {
            _history.push({ role, html, badgeType });
            saveHistory();
        }
        return div;
    }

    function showLoading() {
        removeWelcome();
        const msgs = document.getElementById('fm-messages');
        const div  = document.createElement('div');
        div.className = 'fm-msg ai'; div.id = 'fm-loading';
        div.innerHTML = `<div class="fm-av ai">✨</div>
            <div class="fm-bubble ai"><div class="fm-dots"><span></span><span></span><span></span></div></div>`;
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }
    function hideLoading() {
        const el = document.getElementById('fm-loading');
        if (el) el.remove();
    }

    /* ── 9. Stylist bubble renderer ── */
    function renderStylistBubble(data) {
        const bid    = 'fmst-' + (++_bubbleCounter);
        const shape  = data.face_shape;
        const target = data.target;   // 'hair' | 'glasses' | 'both'
        const samples = data.samples;
        const hasPhoto = data.has_photo;

        stylistState[bid] = { faceShape: shape, selectedHair: '', selectedGlasses: '' };

        let html = `<div style="font-size:12.5px;margin-bottom:10px;">
            Tôi sẽ giúp bạn thử <strong style="color:#f0abfc">${
                target === 'hair' ? 'kiểu tóc' : target === 'glasses' ? 'gọng kính' : 'kiểu tóc và kính'
            }</strong> phù hợp với khuôn mặt <strong style="color:#c4b5fd">${shape}</strong> của bạn!
        </div>`;

        if (!hasPhoto) {
            html += `<div class="fm-no-photo-warn">
                ⚠️ Chưa có ảnh khuôn mặt trong phiên. Vui lòng quay lại
                <a href="/face_shape" style="color:#fbbf24">trang Phân tích khuôn mặt</a>
                và tải ảnh lên trước.
            </div>`;
        } else {
            // Hair section
            if (target === 'hair' || target === 'both') {
                html += buildSampleSection(bid, 'hair', 'Chọn kiểu tóc', samples.hair, shape);
            }
            // Glasses section
            if (target === 'glasses' || target === 'both') {
                html += buildSampleSection(bid, 'glasses', 'Chọn gọng kính', samples.glasses, shape);
            }

            html += `<button class="fm-gen-btn" id="${bid}-genbtn" onclick="fmGenerateStylist('${bid}','${shape}')" disabled>
                ✨ Áp dụng lên ảnh của tôi
            </button>`;
        }

        // Result placeholder
        html += `<div id="${bid}-result" style="display:none"></div>`;

        loadMarked(() => {
            addMsg('ai', html, 'stylist');
            // scroll after render
            const msgs = document.getElementById('fm-messages');
            setTimeout(() => { msgs.scrollTop = msgs.scrollHeight; }, 80);
        });
    }

    function buildSampleSection(bid, type, label, items, shape) {
        if (!items || !items.length) {
            return `<div class="fm-stylist-section"><div class="fm-stylist-label">${label}</div>
                <div class="fm-empty">Chưa có mẫu nào</div></div>`;
        }
        const cards = items.map(it => `
            <div class="fm-sample-card" id="${bid}-${type}-${it.file}"
                 onclick="fmSelectSample('${bid}','${type}','${it.file}')">
                <img src="/face_shape/images/${shape}/${it.file}" alt="${it.name}" loading="lazy">
                <div class="fsc-name">${it.name}</div>
                <div class="fsc-check">✓</div>
            </div>`).join('');
        return `<div class="fm-stylist-section">
            <div class="fm-stylist-label">${label}</div>
            <div class="fm-sample-grid">${cards}</div>
        </div>`;
    }

    /* ── 10. Stylist interaction ── */
    window.fmSelectSample = function (bid, type, file) {
        if (!stylistState[bid]) return;
        // Deselect others of same type
        document.querySelectorAll(`[id^="${bid}-${type}-"]`).forEach(el => {
            el.classList.remove('selected');
        });
        const card = document.getElementById(`${bid}-${type}-${file}`);
        if (card) card.classList.add('selected');

        if (type === 'hair')    stylistState[bid].selectedHair    = file;
        if (type === 'glasses') stylistState[bid].selectedGlasses = file;

        updateGenBtn(bid);
    };

    function updateGenBtn(bid) {
        const st  = stylistState[bid];
        const btn = document.getElementById(`${bid}-genbtn`);
        if (!btn || !st) return;
        btn.disabled = !(st.selectedHair || st.selectedGlasses);
    }

    window.fmGenerateStylist = async function (bid, shape) {
        const st = stylistState[bid];
        if (!st) return;

        const btn     = document.getElementById(`${bid}-genbtn`);
        const resultEl = document.getElementById(`${bid}-result`);
        if (btn) btn.disabled = true;

        // Show mini loading inside bubble
        if (resultEl) {
            resultEl.style.display = 'block';
            resultEl.innerHTML = `<div class="fm-dots" style="margin-top:10px">
                <span></span><span></span><span></span></div>
                <div style="font-size:11px;color:#6b7280;margin-top:4px;">Đang tạo ảnh với AI…</div>`;
        }

        try {
            const res  = await fetch('/api/stylist/generate-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    face_shape:   shape,
                    hair_file:    st.selectedHair,
                    glasses_file: st.selectedGlasses,
                })
            });
            const data = await res.json();

            if (data.error) throw new Error(data.error);

            const url = data.result_url + '?t=' + Date.now();
            if (resultEl) {
                resultEl.innerHTML = `
                    <img class="fm-result-img" src="${url}" alt="Kết quả">
                    <div class="fm-result-actions">
                        <a href="${url}" target="_blank">🔍 Xem lớn</a>
                        <a href="${url}" download>⬇️ Tải xuống</a>
                    </div>`;
            }

            // Mark unread if panel closed
            if (!isOpen) {
                hasUnread = true;
                const badge = document.getElementById('fm-badge');
                if (badge) badge.classList.add('show');
            }

            const msgs = document.getElementById('fm-messages');
            setTimeout(() => { msgs.scrollTop = msgs.scrollHeight; }, 100);

        } catch (e) {
            if (resultEl) {
                resultEl.innerHTML = `<div class="fm-no-photo-warn">❌ ${e.message}</div>`;
            }
            if (btn) btn.disabled = false;
        }
    };

    /* ── 11. Send message ── */
    async function sendMsg() {
        if (busy) return;
        const input = document.getElementById('fm-input');
        const text  = input.value.trim();
        if (!text) return;

        input.value = '';
        input.style.height = 'auto';

        addMsg('user', text);
        showLoading();
        busy = true;
        document.getElementById('fm-send').disabled = true;

        try {
            // Always call /api/stylist/chat – it handles both stylist and normal chat
            const res  = await fetch('/api/stylist/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt:     text,
                    face_shape: faceShape,
                    mode:       mode,
                    page:       pageName,
                })
            });
            const data = await res.json();
            hideLoading();

            loadMarked(() => {
                if (data.error) {
                    addMsg('ai', `❌ ${escHtml(data.error)}`, mode);
                } else if (data.action === 'stylist') {
                    renderStylistBubble(data);
                } else {
                    // Normal chat reply
                    addMsg('ai', parseMd(data.result || ''), mode);
                }

                if (!isOpen) {
                    hasUnread = true;
                    const badge = document.getElementById('fm-badge');
                    if (badge) badge.classList.add('show');
                }
            });

        } catch (err) {
            hideLoading();
            loadMarked(() => addMsg('ai', `❌ Lỗi kết nối: ${err.message}`, mode));
        } finally {
            busy = false;
            document.getElementById('fm-send').disabled = false;
            document.getElementById('fm-input').focus();
        }
    }

    document.getElementById('fm-send').addEventListener('click', sendMsg);
    document.getElementById('fm-input').addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
    });
    document.getElementById('fm-input').addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 90) + 'px';
    });

    /* ── 12. Load marked eagerly + khởi tạo lịch sử chat ── */
    loadMarked(function () {
        // Sau khi marked sẵn sàng mới khôi phục history (cần parseMd)
        initHistory();
    });

})();
