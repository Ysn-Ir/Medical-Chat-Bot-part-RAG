// static/js/common.js

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem("token");
    if (!token) {
        window.location.href = "/login"; 
        return;
    }
    
    // Initialize Sidebar Components
    if(document.getElementById("indexSelect")) {
        loadIndexes();
    }
    
    // Initialize PDF Upload (Sidebar)
    if(document.getElementById("drop-zone")) {
        setupDragAndDrop();
    }
});

// --- UI UTILITIES ---
function showToast(msg, type='info') {
    const container = document.getElementById('toast-container');
    if(!container) return;
    
    const div = document.createElement('div');
    div.className = `toast ${type}`;
    div.textContent = msg;
    container.appendChild(div);
    setTimeout(() => div.remove(), 3000);
}

function logout() {
    localStorage.removeItem("token");
    showToast("Logged out successfully", "success");
    setTimeout(() => {
        window.location.href = "/login";
    }, 1000);
}

// --- INDEX MANAGEMENT (PINECONE) ---
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
    const token = localStorage.getItem("token");
    try {
        const res = await fetch("/set_index", { 
            method: "POST", 
            headers: {"Content-Type":"application/json","Authorization": `Bearer ${token}`}, 
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
    const token = localStorage.getItem("token");
    showToast("Creating index (takes ~20s)...", "info");
    
    try {
        const res = await fetch("/create_index", { 
            method: "POST", 
            headers: {"Content-Type":"application/json","Authorization": `Bearer ${token}`}, 
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

function toggleCreateInput() {
    const el = document.getElementById("createIndexContainer");
    el.style.display = el.style.display === "none" ? "block" : "none";
}

// --- PDF FILE UPLOAD (SIDEBAR) ---
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { ev.preventDefault(); ev.stopPropagation(); });
    });

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
    const token = localStorage.getItem("token");
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    btn.disabled = true; 
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    
    try {
        const res = await fetch("/upload", { 
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` },
            body: formData 
        });
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