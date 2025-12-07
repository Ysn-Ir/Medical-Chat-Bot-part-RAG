// static/js/chat.js
let chatRecorder;
let chatChunks = [];

// Init
document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    if(userInput) {
        userInput.addEventListener("keypress", (e) => { 
            if (e.key === "Enter") sendMessage(); 
        });
    }
});

async function sendMessage() {
    const input = document.getElementById("userInput");
    const text = input.value.trim();
    if (!text) return;

    const sysPrompt = document.getElementById("sysPrompt").value.trim();
    const maxTokens = parseInt(document.getElementById("maxTokens").value);
    const temp = parseFloat(document.getElementById("temperature").value);
    const langSelect = document.getElementById("languageSelect");
    const lang = langSelect ? langSelect.value : "en"; 

    addMessage(text, 'user');
    input.value = "";
    const loaderId = addLoader();
    const token = localStorage.getItem("token");
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json","Authorization": `Bearer ${token}`},
            body: JSON.stringify({ 
                message: text, 
                max_tokens: maxTokens, 
                temperature: temp, 
                system_instruction: sysPrompt || null,
                language: lang
            })
        });
        
        const data = await response.json();
        document.getElementById(loaderId).remove();
        const htmlResponse = marked.parse(data.response);
        addMessage(htmlResponse, 'bot', true);

    } catch (error) {
        document.getElementById(loaderId).remove();
        showToast("Error connecting to AI", 'error');
        console.error(error);
    }
}

async function toggleChatRecording() {
    const btn = document.getElementById("chatMicBtn");
    const input = document.getElementById("userInput");

    if (btn.classList.contains("recording")) {
        if (chatRecorder && chatRecorder.state === "recording") {
            chatRecorder.stop();
        }
        btn.classList.remove("recording");
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>'; 
        return;
    } 
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chatRecorder = new MediaRecorder(stream);
        chatChunks = [];

        chatRecorder.ondataavailable = e => chatChunks.push(e.data);
        
        chatRecorder.onstop = async () => {
            const blob = new Blob(chatChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append("file", blob, "chat.wav");
            const token = localStorage.getItem("token"); 
            try {
                const res = await fetch("/transcribe", { method: "POST",headers:{"Authorization": `Bearer ${token}` }, body: formData });
                const data = await res.json();
                if(data.text) {
                    input.value = data.text;
                } else {
                    showToast("Could not understand audio", "error");
                }
            } catch (e) { 
                showToast("Transcription failed", "error"); 
            }
            
            btn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            btn.style.color = "var(--text-light)";
            stream.getTracks().forEach(track => track.stop());
        };

        chatRecorder.start();
        btn.classList.add("recording");
        btn.innerHTML = '<i class="fa-solid fa-stop"></i>';
        
    } catch (e) {
        console.error(e);
        showToast("Microphone access denied.", "error");
    }
}

// Helpers
function addMessage(text, sender, isHtml = false) {
    const chatbox = document.getElementById("chatbox");
    const div = document.createElement("div");
    div.className = `message ${sender}`;
    const icon = sender === 'user' ? 'fa-user' : 'fa-user-doctor';
    div.innerHTML = `<div class="avatar"><i class="fa-solid ${icon}"></i></div><div class="content">${isHtml ? text : escapeHtml(text)}</div>`;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function addLoader() {
    const id = 'l-' + Date.now();
    const div = document.createElement("div");
    div.className = "message bot";
    div.id = id;
    div.innerHTML = `<div class="avatar"><i class="fa-solid fa-user-doctor"></i></div><div class="content"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
    document.getElementById("chatbox").appendChild(div);
    document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
    return id;
}

function toggleSettings() {
    const el = document.getElementById("settingsPanel");
    el.style.display = el.style.display === "none" ? "block" : "none";
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}