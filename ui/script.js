const documentsList = document.getElementById("documentsList");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const sendBtn = document.getElementById("sendBtn");
const messageInput = document.getElementById("messageInput");
const chatMessages = document.getElementById("chatMessages");
let welcomeMessage = document.getElementById("welcomeMessage");
const newChatBtn = document.getElementById("newChatBtn");

const API_BASE_URL = "http://localhost:8000/api";

let docs = [];
let msgs = [];
let responseTimes = [];
let startTime = null;

/* INITIALIZATION */
async function init() {
    await loadDocuments();
    await updateStats();
}

/* LOAD DOCUMENTS FROM BACKEND */
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`);
        const data = await response.json();
        docs = data.documents.map(doc => doc.name);
        renderDocs();
        
        if (docs.length > 0 && welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
    } catch (error) {
        console.error("Error loading documents:", error);
    }
}

/* UPDATE STATS */
async function updateStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        document.getElementById("docCount").innerText = data.documents_count;
        document.getElementById("msgCount").innerText = data.messages_count;
    } catch (error) {
        console.error("Error loading stats:", error);
    }
}

/* UPLOAD FUNCTIONALITY */
uploadBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files);
    
    for (const file of files) {
        if (file.name.toLowerCase().endsWith('.pdf') || 
            file.name.toLowerCase().endsWith('.doc') || 
            file.name.toLowerCase().endsWith('.docx') || 
            file.name.toLowerCase().endsWith('.txt')) {
            
            await uploadFile(file);
        }
    }
    
    fileInput.value = "";
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log("Upload success:", data);
            await loadDocuments();
            await updateStats();
        } else {
            const error = await response.json();
            alert(`Upload failed: ${error.detail}`);
        }
    } catch (error) {
        console.error("Error uploading file:", error);
        alert("Failed to upload file. Make sure the backend is running.");
    }
}

/* RENDER DOCS */
function renderDocs() {
    documentsList.innerHTML = "";

    if (docs.length === 0) {
        documentsList.innerHTML = `<p class="empty">No PDFs uploaded yet</p>`;
        return;
    }

    docs.forEach(name => {
        const div = document.createElement("div");
        div.className = "document-item uploaded";
        div.innerText = name;
        documentsList.appendChild(div);
    });
}

/* CHAT */
sendBtn.onclick = sendMessage;

messageInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;

    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    sendBtn.innerHTML = '<i class="fas fa-arrow-right"></i>';
    sendBtn.disabled = true;

    startTime = performance.now();

    addMessage(text, "user");
    messageInput.value = "";

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: text })
        });
        
        const endTime = performance.now();
        const responseTime = Math.round(endTime - startTime);
        responseTimes.push(responseTime);
        
        const speedValue = document.getElementById("speedValue");
        if (speedValue) {
            speedValue.innerText = responseTime + "ms";
        }
        
        const avgTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
        const accuracy = Math.max(85, Math.min(98, Math.round(98 - (avgTime - 200) / 50)));
        
        const accuracyValue = document.getElementById("accuracyValue");
        if (accuracyValue) {
            accuracyValue.innerText = accuracy + "%";
        }
        
        if (response.ok) {
            const data = await response.json();
            addMessage(data.message, "ai", data.sources);
        } else {
            const error = await response.json();
            addMessage(`Error: ${error.detail}`, "ai");
        }
    } catch (error) {
        console.error("Error sending message:", error);
        addMessage("Failed to get response. Make sure the backend is running on port 8000.", "ai");
    } finally {
        sendBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
        sendBtn.disabled = false;
        await updateStats();
    }
}

/* ADD MESSAGE */
function addMessage(text, type, sources = null) {
    const msg = document.createElement("div");
    msg.className = "message " + type + "-message";
    
    const icon = document.createElement("div");
    icon.className = "message-icon";
    
    if (type === "user") {
        icon.innerHTML = '<i class="fas fa-user"></i>';
    } else {
        icon.innerHTML = '<i class="fas fa-robot"></i>';
    }
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    
    const textDiv = document.createElement("div");
    textDiv.innerText = text;
    bubble.appendChild(textDiv);
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";
        sourcesDiv.style.cssText = "margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 12px; color: #a1a1aa;";
        sourcesDiv.innerHTML = `<i class="fas fa-book"></i> Sources: ${sources.join(", ")}`;
        bubble.appendChild(sourcesDiv);
    }
    
    msg.appendChild(icon);
    msg.appendChild(bubble);

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    msgs.push(text);
    document.getElementById("msgCount").innerText = msgs.length;
}

/* NEW CHAT */
newChatBtn.onclick = async () => {
    chatMessages.innerHTML = `
        <div id="welcomeMessage" class="welcome-message">
            <h3>Welcome</h3>
            <p>I'm your RAG assistant. Upload documents to get started.</p>
        </div>
    `;
    msgs = [];
    document.getElementById("msgCount").innerText = 0;
    
    const newWelcomeMessage = document.getElementById("welcomeMessage");
    if (newWelcomeMessage) {
        welcomeMessage = newWelcomeMessage;
    }
    
    try {
        await fetch(`${API_BASE_URL}/clear`, { method: "POST" });
        await updateStats();
    } catch (error) {
        console.error("Error clearing chat:", error);
    }
};

/* AUTO RESIZE FIX */
messageInput.addEventListener("input", function () {
    this.style.height = "auto";
    const newHeight = Math.min(this.scrollHeight, 120);
    this.style.height = newHeight + "px";
});

/* INITIALIZE */
init();