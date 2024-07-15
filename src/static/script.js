document.addEventListener('DOMContentLoaded', (event) => {
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileName');
    const uploadButton = document.getElementById('uploadButton');

    // Handle file selection
    fileInput.addEventListener('change', function() {
        const file = fileInput.files[0];
        if (file) {
            fileNameDisplay.textContent = `Selected file: ${file.name}`;
        } else {
            fileNameDisplay.textContent = 'No file selected';
        }
    });

    // Handle file upload
    uploadButton.addEventListener('click', function() {
        const file = fileInput.files[0];
        if (file) {
            console.log('Uploading file:', file.name);
            fileNameDisplay.textContent = `Uploading file: ${file.name}`;

            const form = document.getElementById('uploadForm');
            const formData = new FormData(form);

            fetch('/upload', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.text())
            .then(data => {
                document.open();
                document.write(data);
                document.close();
            })
            .catch(error => {
                console.error('Error uploading file:', error);
                fileNameDisplay.textContent = `Error uploading file: ${file.name}`;
            });
        } else {
            alert('Please select a file to upload');
        }
    });
});
