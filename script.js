let currentFile = '';

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('pdf');
    formData.append('pdf', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    if (data.filename) {
        currentFile = data.filename;
        document.getElementById('fileInfo').style.display = 'block';
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
    const output = document.getElementById('output');
    output.innerHTML = `
        <h2 class="question-title">What</h2>
        <p>${data.what}</p>
        <h2 class="question-title">Why</h2>
        <p>${data.why}</p>
        <h2 class="question-title">How</h2>
        <p>${data.how}</p>
        <h2 class="question-title">Summary</h2>
        <p>${data.summary}</p>
    `;
    document.getElementById('results').style.display = 'block';
}

async function importantPoints() {
    const topicInput = document.getElementById('topic');
    const pageInput = document.getElementById('page');
    if (!topicInput || !pageInput) return;

    const topic = topicInput.value;
    const page = pageInput.value;

    const formData = new FormData();
    formData.append('filename', currentFile);
    formData.append('topic', topic);
    formData.append('page', page);

    const res = await fetch('/important-points', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    document.getElementById('important-points').textContent = data.points;
    document.getElementById('results').style.display = 'block';
}

async function generateQuiz() {
    const topicInput = document.getElementById('topic');
    const pageInput = document.getElementById('page');
    if (!topicInput || !pageInput) return;

    const topic = topicInput.value;
    const page = pageInput.value;

    const formData = new FormData();
    formData.append('filename', currentFile);
    formData.append('topic', topic);
    formData.append('page', page);

    const res = await fetch('/generate-quiz', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    document.getElementById('quiz-output').textContent = data.quiz;
    document.getElementById('results').style.display = 'block';
}