// --- GLOBAL STATE ---
let chatRecorder;
let chatChunks = [];
let diagnosisRecorder;
let diagnosisChunks = [];
let audioBlobs = { cough: null, breath: null, speech: null };

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    loadIndexes();
    setupDragAndDrop();
    document.getElementById('userInput').addEventListener("keypress", (e) => { 
        if (e.key === "Enter") sendMessage(); 
    });
});

// ==========================================
//  1. CHAT & VOICE-TO-TEXT LOGIC
// ==========================================

async function sendMessage() {
    const input = document.getElementById("userInput");
    const text = input.value.trim();
    if (!text) return;

    // 1. Get the current settings
    const sysPrompt = document.getElementById("sysPrompt").value.trim();
    const maxTokens = parseInt(document.getElementById("maxTokens").value);
    const temp = parseFloat(document.getElementById("temperature").value);
    
    // 2. CRITICAL: Get the selected language
    // Make sure your dropdown in HTML has id="languageSelect"
    const langSelect = document.getElementById("languageSelect");
    const lang = langSelect ? langSelect.value : "en"; 

    addMessage(text, 'user');
    input.value = "";
    const loaderId = addLoader();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                message: text, 
                max_tokens: maxTokens, 
                temperature: temp, 
                system_instruction: sysPrompt || null,
                language: lang // <--- THIS MUST BE SENT
            })
        });
        
        const data = await response.json();
        document.getElementById(loaderId).remove();
        
        // Parse Markdown
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

    // STOP RECORDING
    if (btn.classList.contains("recording")) {
        if (chatRecorder && chatRecorder.state === "recording") {
            chatRecorder.stop();
        }
        btn.classList.remove("recording");
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>'; 
        return;
    } 
    
    // START RECORDING
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chatRecorder = new MediaRecorder(stream);
        chatChunks = [];

        chatRecorder.ondataavailable = e => chatChunks.push(e.data);
        
        chatRecorder.onstop = async () => {
            const blob = new Blob(chatChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append("file", blob, "chat.wav");

            try {
                const res = await fetch("/transcribe", { method: "POST", body: formData });
                const data = await res.json();
                if(data.text) {
                    input.value = data.text;
                } else {
                    showToast("Could not understand audio", "error");
                }
            } catch (e) { 
                showToast("Transcription failed", "error"); 
            }
            
            // Reset UI
            btn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            btn.style.color = "var(--text-light)";
            stream.getTracks().forEach(track => track.stop());
        };

        chatRecorder.start();
        btn.classList.add("recording");
        btn.innerHTML = '<i class="fa-solid fa-stop"></i>';
        
    } catch (e) {
        console.error(e);
        showToast("Microphone access denied. Check permissions.", "error");
    }
}

// ==========================================
//  2. MEDICAL DIAGNOSIS (MODAL)
// ==========================================

function openDiagnosisModal() {
    const modal = document.getElementById("voiceModal");
    modal.style.display = "flex";
    
    // Reset State
    audioBlobs = { cough: null, breath: null, speech: null };
    ['cough', 'breath', 'speech'].forEach(type => {
        const card = document.getElementById(`card-${type}`);
        card.classList.remove('recorded', 'recording');
        document.getElementById(`status-${type}`).textContent = "Click to Record";
    });
    document.getElementById("analyzeBtn").disabled = true;
    document.getElementById("analysisResult").style.display = "none";
}

function closeVoiceModal() {
    document.getElementById("voiceModal").style.display = "none";
}

async function recordSample(type) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        diagnosisRecorder = new MediaRecorder(stream);
        diagnosisChunks = [];
        
        const card = document.getElementById(`card-${type}`);
        const status = document.getElementById(`status-${type}`);
        
        // UI Start
        card.classList.add("recording");
        status.textContent = "Recording (4s)...";

        diagnosisRecorder.ondataavailable = e => diagnosisChunks.push(e.data);
        
        diagnosisRecorder.onstop = () => {
            audioBlobs[type] = new Blob(diagnosisChunks, { type: 'audio/wav' });
            
            // UI Stop
            card.classList.remove("recording");
            card.classList.add("recorded");
            status.textContent = "Recorded ✓";
            
            stream.getTracks().forEach(track => track.stop());
            
            // Enable Analyze button if all 3 are ready
            if (audioBlobs.cough && audioBlobs.breath && audioBlobs.speech) {
                document.getElementById("analyzeBtn").disabled = false;
            }
        };

        diagnosisRecorder.start();
        
        // Auto stop after 4 seconds (Standardized)
        setTimeout(() => {
            if (diagnosisRecorder.state === "recording") {
                diagnosisRecorder.stop();
            }
        }, 4000);

    } catch (e) {
        showToast("Microphone error. Ensure permission is granted.", "error");
    }
}

async function submitDiagnosis() {
    const btn = document.getElementById("analyzeBtn");
    const resultDiv = document.getElementById("analysisResult");
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    
    const formData = new FormData();
    formData.append("cough", audioBlobs.cough, "cough.wav");
    formData.append("breath", audioBlobs.breath, "breath.wav");
    formData.append("speech", audioBlobs.speech, "speech.wav");

    try {
        const res = await fetch("/diagnose", { method: "POST", body: formData });
        const data = await res.json();
        
        resultDiv.style.display = "block";
        
        if (data.status === "HEALTHY") {
            resultDiv.style.background = "#ecfdf5";
            resultDiv.style.color = "#047857";
            resultDiv.style.border = "1px solid #10b981";
            resultDiv.innerHTML = `
                <div style="font-size:1.1rem; font-weight:600; margin-bottom:5px;">🟢 Healthy</div>
                <div>${data.details}</div>
                <div style="font-size:0.8rem; margin-top:5px; opacity:0.8;">Sickness Probability: ${(data.sickness_prob * 100).toFixed(1)}%</div>
            `;
        } else {
            resultDiv.style.background = "#fef2f2";
            resultDiv.style.color = "#b91c1c";
            resultDiv.style.border = "1px solid #ef4444";
            
            let html = `
                <div style="font-size:1.1rem; font-weight:600; margin-bottom:5px;">🔴 ${data.status}</div>
                <div>${data.details}</div>
            `;
            if (data.covid_prob) {
                html += `<div style="font-size:0.8rem; margin-top:5px; opacity:0.8;">COVID Probability: ${(data.covid_prob * 100).toFixed(1)}%</div>`;
            }
            resultDiv.innerHTML = html;
        }

    } catch (e) {
        showToast("Analysis failed. Check server logs.", "error");
    }
    
    btn.disabled = false;
    btn.innerHTML = "Analyze Samples";
}

// ==========================================
//  3. INDEX MANAGEMENT (Pinecone)
// ==========================================

async function loadIndexes() {
    const select = document.getElementById("indexSelect");
    const refreshBtn = document.getElementById("refreshIndexBtn");
    
    refreshBtn.innerHTML = '<i class="fa-solid fa-sync fa-spin"></i>';

    try {
        const res = await fetch("/indexes");
        const data = await res.json();
        
        select.innerHTML = "";
        data.indexes.forEach(idx => {
            const opt = document.createElement("option");
            opt.value = idx; 
            opt.text = idx;
            if (idx === data.current_index) opt.selected = true;
            select.appendChild(opt);
        });
    } catch(e) {
        console.warn("Could not load indexes");
    } finally {
        refreshBtn.innerHTML = '<i class="fa-solid fa-sync"></i>';
    }
}

async function switchIndex(name) {
    if (!name) return;
    showToast(`Switching to ${name}...`, 'info');
    
    try {
        const res = await fetch("/set_index", { 
            method: "POST", 
            headers: {"Content-Type":"application/json"}, 
            body: JSON.stringify({index_name: name})
        });
        if (res.ok) showToast("Connected!", "success");
        else showToast("Failed to switch", "error");
    } catch (e) {
        showToast("Network error", "error");
    }
}

async function createNewIndex() {
    const nameInput = document.getElementById("newIndexName");
    const name = nameInput.value.trim();
    if (!name) return showToast("Enter a name", "error");

    showToast("Creating index (takes ~20s)...", "info");
    
    try {
        const res = await fetch("/create_index", { 
            method: "POST", 
            headers: {"Content-Type":"application/json"}, 
            body: JSON.stringify({index_name: name})
        });
        
        if (res.ok) {
            showToast("Index created!", "success");
            toggleCreateInput();
            nameInput.value = "";
            loadIndexes();
        } else {
            showToast("Failed to create index", "error");
        }
    } catch (e) {
        showToast("Error creating index", "error");
    }
}

// ==========================================
//  4. FILE UPLOAD LOGIC
// ==========================================

function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { ev.preventDefault(); ev.stopPropagation(); });
    });

    ['dragenter', 'dragover'].forEach(e => dropZone.classList.add('dragover'));
    ['dragleave', 'drop'].forEach(e => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        document.getElementById('fileInput').files = files;
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('uploadBtn').disabled = false;
    }
}

async function uploadFile() {
    const btn = document.getElementById("uploadBtn");
    const file = document.getElementById("fileInput").files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    btn.disabled = true; 
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    
    try {
        const res = await fetch("/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (res.ok) {
            showToast(`Success! ${data.chunks_added} chunks added.`, "success");
        } else {
            showToast(data.detail, "error");
        }
    } catch (e) {
        showToast("Upload failed", "error");
    }
    
    btn.disabled = false; 
    btn.innerHTML = "Analyze Document";
}

// ==========================================
//  5. UI UTILITIES
// ==========================================

function addMessage(text, sender, isHtml = false) {
    const chatbox = document.getElementById("chatbox");
    const div = document.createElement("div");
    div.className = `message ${sender}`;
    
    const icon = sender === 'user' ? 'fa-user' : 'fa-user-doctor';
    
    div.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="content">${isHtml ? text : escapeHtml(text)}</div>
    `;
    
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function addLoader() {
    const id = 'l-' + Date.now();
    const div = document.createElement("div");
    div.className = "message bot";
    div.id = id;
    div.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-user-doctor"></i></div>
        <div class="content">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
    `;
    document.getElementById("chatbox").appendChild(div);
    document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
    return id;
}

function toggleSettings() {
    const el = document.getElementById("settingsPanel");
    el.style.display = el.style.display === "none" ? "block" : "none";
}

function toggleCreateInput() {
    const el = document.getElementById("createIndexContainer");
    el.style.display = el.style.display === "none" ? "block" : "none";
}

function showToast(msg, type='info') {
    const container = document.getElementById('toast-container');
    const div = document.createElement('div');
    div.className = `toast ${type}`;
    div.textContent = msg;
    container.appendChild(div);
    setTimeout(() => div.remove(), 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}