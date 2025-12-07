// Voice Logic (Extracted from script.js)
let diagnosisRecorder;
let diagnosisChunks = [];
let audioBlobs = { cough: null, breath: null, speech: null };

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
    const token = localStorage.getItem("token"); // Get Token
    try {
        const res = await fetch("/diagnose", { method: "POST",headers:{"Authorization": `Bearer ${token}`},  body: formData });
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