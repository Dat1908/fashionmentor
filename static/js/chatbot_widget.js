/**
 * FashionMentor AI – Floating Chatbot Widget
 * Tự inject vào trang, không cần thêm HTML thủ công.
 */
(function () {
    'use strict';

    /* ── 1. Inject CSS ── */
    const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#fm-fab {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.55);
    transition: transform 0.25s, box-shadow 0.25s;
    font-size: 22px;
    color: #fff;
    user-select: none;
}
#fm-fab:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 28px rgba(124, 58, 237, 0.7);
}
#fm-fab .fm-badge {
    position: absolute;
    top: -4px;
    right: -4px;
    background: #ec4899;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid #07070f;
    opacity: 0;
    transition: opacity 0.2s;
}
#fm-fab .fm-badge.show { opacity: 1; }

#fm-panel {
    position: fixed;
    bottom: 96px;
    right: 28px;
    z-index: 9998;
    width: 370px;
    height: 520px;
    border-radius: 20px;
    background: rgba(10, 10, 20, 0.97);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(124,58,237,0.2);
    display: flex;
    flex-direction: column;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
    transform: scale(0.85) translateY(20px);
    transform-origin: bottom right;
    opacity: 0;
    pointer-events: none;
    transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1), opacity 0.22s ease;
}
#fm-panel.open {
    transform: scale(1) translateY(0);
    opacity: 1;
    pointer-events: all;
}

/* Panel Header */
.fm-header {
    padding: 14px 16px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    background: rgba(255,255,255,0.025);
}
.fm-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.fm-avatar-hd {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.fm-header-info .fm-title {
    font-size: 13.5px; font-weight: 600; color: #f1f5f9;
}
.fm-header-info .fm-sub {
    font-size: 11px; color: #6b7280; margin-top: 1px;
}
.fm-header-actions { display: flex; align-items: center; gap: 6px; }

/* Mode toggle */
.fm-mode-toggle {
    display: flex;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 30px;
    padding: 2px;
    gap: 1px;
}
.fm-mode-btn {
    padding: 4px 11px;
    border-radius: 30px; border: none;
    font-size: 11px; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    background: transparent; color: #9ca3af;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}
.fm-mode-btn.active {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: #fff;
    box-shadow: 0 1px 10px rgba(124,58,237,0.4);
}
.fm-close-btn {
    width: 28px; height: 28px;
    border-radius: 50%; border: none;
    background: rgba(255,255,255,0.07);
    color: #9ca3af; font-size: 14px;
    cursor: pointer; display: flex;
    align-items: center; justify-content: center;
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

/* Welcome inside panel */
.fm-welcome {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; gap: 10px; opacity: 0.7;
    padding: 20px;
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
    max-width: 82%; padding: 9px 12px;
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

/* Input area */
.fm-input-area {
    padding: 10px 12px 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    flex-shrink: 0;
    background: rgba(255,255,255,0.015);
}
.fm-input-box {
    display: flex; gap: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 7px 8px 7px 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.fm-input-box:focus-within {
    border-color: rgba(168,85,247,0.5);
    box-shadow: 0 0 0 3px rgba(168,85,247,0.1);
}
.fm-input-box textarea {
    flex: 1; background: none; border: none; outline: none;
    font-family: 'Inter',sans-serif; font-size: 12.5px;
    color: #e2e8f0; resize: none;
    max-height: 90px; min-height: 20px; line-height: 1.5;
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

    /* ── 2. Detect current page context ── */
    const path    = window.location.pathname;
    let pageName  = '';
    let topicName = 'thời trang';
    const m = path.match(/\/(face_shape|body_shape|personal_color)\/(.+)/);
    if (m) {
        pageName  = m[2].replace(/-/g, '_');
        topicName = pageName.replace(/_/g, ' ');
    }

    /* ── 3. Build DOM ── */
    // FAB button
    const fab = document.createElement('button');
    fab.id = 'fm-fab';
    fab.title = 'Mở FashionMentor AI';
    fab.innerHTML = '💬<span class="fm-badge" id="fm-badge">1</span>';
    document.body.appendChild(fab);

    // Panel
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
        <p>Xin chào! Tôi là trợ lý AI thời trang.<br>
        ${pageName ? `Tôi sẽ tư vấn dựa trên nội dung trang <b style="color:#c4b5fd">${topicName}</b> này.` : 'Hãy hỏi tôi về thời trang!'}</p>
    </div>
</div>
<div class="fm-input-area">
    <div class="fm-input-box">
        <textarea id="fm-input" placeholder="Hỏi về thời trang..." rows="1"></textarea>
        <button class="fm-send" id="fm-send" title="Gửi">➤</button>
    </div>
    <div class="fm-hint" id="fm-hint">💬 Chat · Gemini 2.5 Flash</div>
</div>`;
    document.body.appendChild(panel);

    /* ── 4. State ── */
    let isOpen   = false;
    let mode     = 'chat';
    let busy     = false;
    let hasUnread = false;

    /* ── 5. Wire up marked.js (lazy load) ── */
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
        return text.replace(/\n/g, '<br>');
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
            m2 === 'chat'
            ? '💬 Chat · Gemini 2.5 Flash'
            : '🔍 Web Search · Gemini 2.5 Pro';
    };

    /* ── 8. Messages ── */
    function escHtml(t) {
        return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    }

    function addMsg(role, content) {
        const welcome = document.getElementById('fm-welcome');
        if (welcome) welcome.remove();

        const msgs = document.getElementById('fm-messages');
        const div  = document.createElement('div');
        div.className = `fm-msg ${role}`;

        if (role === 'ai') {
            const badgeCls = mode === 'search' ? 'search' : '';
            const badgeTxt = mode === 'search' ? '🔍 Search' : '💬 Chat';
            div.innerHTML = `
                <div class="fm-av ai">✨</div>
                <div class="fm-bubble ai">
                    <div class="fm-badge-mode ${badgeCls}">${badgeTxt}</div>
                    ${parseMd(content)}
                </div>`;
        } else {
            div.innerHTML = `
                <div class="fm-av user">👤</div>
                <div class="fm-bubble user">${escHtml(content)}</div>`;
        }

        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }

    function showLoading() {
        const welcome = document.getElementById('fm-welcome');
        if (welcome) welcome.remove();
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

    /* ── 9. Send ── */
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
            const res  = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, mode: mode, page: pageName })
            });
            const data = await res.json();
            hideLoading();

            loadMarked(() => {
                if (data.error) {
                    addMsg('ai', '❌ Lỗi: ' + data.error);
                } else {
                    addMsg('ai', data.result);
                    if (!isOpen) {
                        hasUnread = true;
                        const badge = document.getElementById('fm-badge');
                        if (badge) badge.classList.add('show');
                    }
                }
            });
        } catch (err) {
            hideLoading();
            addMsg('ai', '❌ Lỗi kết nối: ' + err.message);
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

    /* ── 10. Load marked eagerly ── */
    loadMarked(function () {});

})();
