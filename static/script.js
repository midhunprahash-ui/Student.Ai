let currentFile = '';

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const file = document.getElementById('pdf').files[0];
    formData.append('pdf', file);

    const res = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    if (data.filename) {
        currentFile = data.filename;
        document.getElementById('fileInfo').style.display = 'block';
        alert('File uploaded successfully!');
    } else {
        alert('Upload failed.');
    }
});

async function analyze() {
    const topic = document.getElementById('topic').value;
    const page = document.getElementById('page').value;
    const formData = new FormData();
    formData.append('topic', topic);
    formData.append('page', page);
    formData.append('filename', currentFile);

    const res = await fetch('/analyze', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    document.getElementById('results').style.display = 'block';
    document.getElementById('output').innerHTML =
        `🔎 <strong>What:</strong> ${data.what}\n\n❓ <strong>Why:</strong> ${data.why}\n\n⚙️ <strong>How:</strong> ${data.how}\n\n📄 <strong>Summary:</strong> ${data.summary}`;
}

async function importantPoints() {
    const page = document.getElementById('page').value;
    const formData = new FormData();
    formData.append('page', page);
    formData.append('filename', currentFile);

    const res = await fetch('/important-points', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    document.getElementById('results').style.display = 'block';
    document.getElementById('output').innerHTML = `📌 <strong>Important Points:</strong>\n${data.points}`;
}

async function generateQuiz() {
    const page = document.getElementById('page').value;
    const formData = new FormData();
    formData.append('page', page);
    formData.append('filename', currentFile);

    const res = await fetch('/generate-quiz', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    document.getElementById('results').style.display = 'block';
    document.getElementById('output').innerHTML = `📝 <strong>Quiz:</strong>\n${data.quiz}`;
}
