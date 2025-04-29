function uploadVideo() {
    const fileInput = document.getElementById("videoUpload");
    if (fileInput.files.length === 0) {
        alert("Please select a video file!");
        return;
    }

    const videoFile = fileInput.files[0];
    const videoURL = URL.createObjectURL(videoFile);

    localStorage.setItem("uploadedVideo", videoURL);
    window.location.href = "analyze.html";
}

function loadVideo() {
    const videoPlayer = document.getElementById("videoPlayer");
    const uploadedVideo = localStorage.getItem("uploadedVideo");

    if (uploadedVideo) {
        videoPlayer.src = uploadedVideo;
    } else {
        alert("No video found! Please upload a video.");
        window.location.href = "index.html";
    }
}

function goBack() {
    localStorage.removeItem("uploadedVideo");
    window.location.href = "index.html";
}

// Show preview before upload
document.getElementById("videoUpload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        const videoPreview = document.getElementById("videoPreview");
        videoPreview.src = URL.createObjectURL(file);
        videoPreview.classList.remove("d-none");
    }
});
