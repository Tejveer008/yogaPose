let video;
let pose;
let skeleton;
let pose_names = ["MOUNTAIN", "GODDESS", "GARLAND", "PLANK"];
let pose_images = {
    "MOUNTAIN": "image1.png",
    "GODDESS": "image2.jpeg",
    "GARLAND": "image3.jpg",
    "PLANK": "image4.jpg"
};
let posesDropdown;
let selected_pose = "none";
let poseLabel = "";
let poseNet;
let knn;

let count = { correct: 0, total: 0 }; // Counter for accuracy calculation
const DATA_PATH = "./data.json";

function setup() {
    let cv = createCanvas(320, 240); // Reduced canvas size for better performance
    cv.parent("imager");
    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide();

    // Dropdown setup
    posesDropdown = document.getElementById("poses_dropdown");

    // Initialize PoseNet model
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on("pose", getPoses);

    // Initialize KNN Classifier
    knn = ml5.KNNClassifier();
}

function image_maker() {
    let posesDropdownVal = posesDropdown.value;
    if (pose_images[posesDropdownVal]) {
        document.getElementById("pose_img").src = pose_images[posesDropdownVal];
    } else {
        document.getElementById("pose_img").src = ""; // Clear image if invalid selection
    }
}

function modelLoaded() {
    console.log("PoseNet model loaded!");
    knn.load(DATA_PATH, networkLoaded);
}

function networkLoaded() {
    console.log("KNN loaded!");
    classifyPose();
}

function classifyPose() {
    if (pose) {
        const poseArray = pose.keypoints.map((p) => [p.score, p.position.x, p.position.y]);
        knn.classify(poseArray, gotResult);
    } else {
        setTimeout(classifyPose, 500); // Retry if no pose detected, reduced interval for responsiveness
    }
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }
    if (results) {
        count.total++;
        poseLabel = pose_names[parseInt(results.label)];

        if (posesDropdown.value === poseLabel) {
            count.correct++;
        }
        classifyPose(); // Continue classification
    }
}

function getPoses(poses) {
    if (poses.length > 0) {
        pose = poses[0].pose;
        skeleton = poses[0].skeleton;
    }
}

function drawKeypoints() {
    if (pose) {
        for (let keypoint of pose.keypoints) {
            if (keypoint.score > 0.2) {
                fill(255, 0, 0);
                noStroke();
                ellipse(keypoint.position.x, keypoint.position.y, 5, 5); // Reduced size for better performance
            }
        }
    }
}

function drawSkeleton() {
    if (pose) {
        for (let [partA, partB] of skeleton) {
            stroke(255, 0, 0);
            line(
                partA.position.x,
                partA.position.y,
                partB.position.x,
                partB.position.y
            );
        }
    }
}

function draw() {
    if (selected_pose !== "none") {
        image(video, 0, 0, width, height);

        drawKeypoints();
        drawSkeleton();

        fill(0, 255, 0);
        textSize(20); // Reduced text size for better visibility on smaller screens

        if (posesDropdown.value !== poseLabel) {
            text("INCORRECT POSE", width / 4, height - 10);
        } else {
            text(`${poseLabel} POSE`, width / 4, height - 10);
        }
    }
    selected_pose = posesDropdown.value;
}

function end() {
    console.log(count);
    console.log("Accuracy: ", ((count.correct / count.total) * 100).toFixed(2), "%");
    count = { correct: 0, total: 0 };
    video.stop();
    video.remove();
}
