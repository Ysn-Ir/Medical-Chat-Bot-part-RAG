// Vision Logic (Extracted from script.js)

document.addEventListener('DOMContentLoaded', () => {
    const xrayDrop = document.getElementById('xray-drop-zone');
    const xrayInput = document.getElementById('xrayInput');

    if (xrayDrop && xrayInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
            xrayDrop.addEventListener(e, (ev) => { ev.preventDefault(); ev.stopPropagation(); });
        });

        xrayDrop.addEventListener('drop', (e) => handleXrayFiles(e.dataTransfer.files));
        xrayInput.addEventListener('change', (e) => handleXrayFiles(e.target.files));
    }
});

function handleXrayFiles(files) {
    if (files.length > 0) {
        document.getElementById('xrayInput').files = files;
        document.getElementById('xray-name').textContent = files[0].name;
        document.getElementById('runVisionBtn').disabled = false;
    }
}

async function submitXray() {
    const file = document.getElementById("xrayInput").files[0];
    if(!file) return;

    const btn = document.getElementById("runVisionBtn");
    const loading = document.getElementById("visionLoading");
    const imgOutput = document.getElementById("visionOutput");
    const resultDiv = document.getElementById("visionResult");

    // UI Loading
    btn.disabled = true;
    resultDiv.style.display = "block";
    loading.style.display = "block";
    imgOutput.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    const token = localStorage.getItem("token");

    try {
        const res = await fetch("/analyze_xray", {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` },
            body: formData
        });
        
        const data = await res.json();
        
        loading.style.display = "none";
        
        if (data.status === "success") {
            imgOutput.src = `data:image/png;base64,${data.image_base64}`;
            imgOutput.style.display = "block";
            showToast(`Detection: ${data.top_finding}`, "success");
        } else {
            showToast("Analysis failed: " + data.message, "error");
        }

    } catch (e) {
        loading.style.display = "none";
        showToast("Server error during analysis", "error");
    }
    
    btn.disabled = false;
}