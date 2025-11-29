document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('drop-zone');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadBtn = document.getElementById('uploadBtn');
    const userInput = document.getElementById('userInput');

    // Drag & Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleDrop(e) {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            fileInput.files = files; // Sync with input
            fileNameDisplay.textContent = files[0].name;
            uploadBtn.disabled = false;
        }
    }

    // Enter key support
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
});

// Toast Notification System
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const btn = document.getElementById("uploadBtn");
    
    if (!fileInput.files[0]) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();
        
        if (response.ok) {
            showToast(`Success! ${data.chunks_added} chunks indexed.`, 'success');
        } else {
            showToast(`Error: ${data.detail}`, 'error');
        }
    } catch (e) {
        showToast("Upload failed due to connection error.", 'error');
    }
    
    btn.disabled = false;
    btn.textContent = "Analyze Document";
}

async function sendMessage() {
    const input = document.getElementById("userInput");
    const chatbox = document.getElementById("chatbox");
    const text = input.value.trim();

    if (!text) return;

    // Add User Message
    addMessage(text, 'user');
    input.value = "";

    // Add Loading Indicator
    const loaderId = addLoader();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        
        // Remove Loader
        document.getElementById(loaderId).remove();
        
        // Add Bot Message (Parsed with Marked.js for formatting)
        const formattedResponse = marked.parse(data.response);
        addMessage(formattedResponse, 'bot', true);

    } catch (error) {
        document.getElementById(loaderId).remove();
        showToast("Failed to get response from AI.", 'error');
        addMessage("Sorry, I encountered an error connecting to the server.", 'bot');
    }
}

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
    const chatbox = document.getElementById("chatbox");
    const id = 'loader-' + Date.now();
    const div = document.createElement("div");
    div.className = `message bot`;
    div.id = id;
    div.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-user-doctor"></i></div>
        <div class="content">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
    `;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
    return id;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}