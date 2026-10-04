// =========================================================
// AI FITNESS TRACKER
// PHONE SENSOR DATA COLLECTOR
// =========================================================


// ---------------------------------------------------------
// Constants
// ---------------------------------------------------------

const G_TO_MS2 = 9.80665;


// ---------------------------------------------------------
// DOM elements
// ---------------------------------------------------------

const statusElement =
    document.getElementById("status");

const sampleCountElement =
    document.getElementById("sampleCount");

const elapsedTimeElement =
    document.getElementById("elapsedTime");


const accXElement =
    document.getElementById("accX");

const accYElement =
    document.getElementById("accY");

const accZElement =
    document.getElementById("accZ");


const gyroXElement =
    document.getElementById("gyroX");

const gyroYElement =
    document.getElementById("gyroY");

const gyroZElement =
    document.getElementById("gyroZ");


const exerciseElement =
    document.getElementById("exercise");

const intensityElement =
    document.getElementById("intensity");


const startBtn =
    document.getElementById("startBtn");

const stopBtn =
    document.getElementById("stopBtn");

const downloadBtn =
    document.getElementById("downloadBtn");

const clearBtn =
    document.getElementById("clearBtn");


// ---------------------------------------------------------
// State
// ---------------------------------------------------------

let recording = false;

let startTime = null;

let timerInterval = null;

let sampleCount = 0;

let sensorData = [];


// ---------------------------------------------------------
// Sensor support
// ---------------------------------------------------------

function checkSensorSupport() {

    if (!("DeviceMotionEvent" in window)) {

        alert(
            "Device motion sensors are not supported by this browser."
        );

        return false;
    }

    return true;
}


// ---------------------------------------------------------
// Request sensor permission
// ---------------------------------------------------------

async function requestPermission() {

    try {

        // Some browsers require explicit permission
        // from a user gesture.

        if (
            typeof DeviceMotionEvent.requestPermission
            === "function"
        ) {

            const permission =
                await DeviceMotionEvent.requestPermission();

            if (permission !== "granted") {

                alert(
                    "Sensor permission was denied."
                );

                return false;
            }
        }

        return true;

    } catch (error) {

        console.error(
            "Permission error:",
            error
        );

        alert(
            "Unable to access motion sensors."
        );

        return false;
    }
}


// ---------------------------------------------------------
// Safe number helper
// ---------------------------------------------------------

function safeValue(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(value)
    ) {
        return 0;
    }

    return Number(value);
}


// ---------------------------------------------------------
// Motion handler
// ---------------------------------------------------------

function handleMotion(event) {

    if (!recording) {
        return;
    }


    // -----------------------------------------------------
    // Acceleration
    // -----------------------------------------------------

    let acceleration =
        event.acceleration;


    // Some devices/browser combinations may provide
    // accelerationIncludingGravity instead.

    if (
        !acceleration ||
        (
            acceleration.x === null &&
            acceleration.y === null &&
            acceleration.z === null
        )
    ) {

        acceleration =
            event.accelerationIncludingGravity;
    }


    if (!acceleration) {
        return;
    }


    // Browser acceleration is m/s².
    // Our existing dataset uses g for accelerometer.

    let axMs2 =
        safeValue(acceleration.x);

    let ayMs2 =
        safeValue(acceleration.y);

    let azMs2 =
        safeValue(acceleration.z);


    let ax =
        axMs2 / G_TO_MS2;

    let ay =
        ayMs2 / G_TO_MS2;

    let az =
        azMs2 / G_TO_MS2;


    // -----------------------------------------------------
    // Gyroscope
    // -----------------------------------------------------

    const rotation =
        event.rotationRate;


    let gx = 0;
    let gy = 0;
    let gz = 0;


    if (rotation) {

        gx =
            safeValue(rotation.alpha);

        gy =
            safeValue(rotation.beta);

        gz =
            safeValue(rotation.gamma);
    }


    // -----------------------------------------------------
    // Timestamp
    // -----------------------------------------------------

    const now =
        Date.now();

    const elapsed =
        (now - startTime) / 1000;


    // -----------------------------------------------------
    // Selected metadata
    // -----------------------------------------------------

    const exercise =
        exerciseElement.value;

    const intensity =
        intensityElement.value;


    // -----------------------------------------------------
    // Save sample
    // -----------------------------------------------------

    const sample = {

        epoch_ms: now,

        elapsed_s:
            Number(elapsed.toFixed(4)),

        exercise:
            exercise,

        intensity:
            intensity,

        acc_x_g:
            Number(ax.toFixed(6)),

        acc_y_g:
            Number(ay.toFixed(6)),

        acc_z_g:
            Number(az.toFixed(6)),

        gyro_x_dps:
            Number(gx.toFixed(6)),

        gyro_y_dps:
            Number(gy.toFixed(6)),

        gyro_z_dps:
            Number(gz.toFixed(6)),

        interval_ms:
            safeValue(event.interval)

    };


    sensorData.push(sample);

    sampleCount++;


    // -----------------------------------------------------
    // Update UI
    // -----------------------------------------------------

    sampleCountElement.textContent =
        sampleCount;


    elapsedTimeElement.textContent =
        elapsed.toFixed(1);


    accXElement.textContent =
        ax.toFixed(3);

    accYElement.textContent =
        ay.toFixed(3);

    accZElement.textContent =
        az.toFixed(3);


    gyroXElement.textContent =
        gx.toFixed(3);

    gyroYElement.textContent =
        gy.toFixed(3);

    gyroZElement.textContent =
        gz.toFixed(3);
}


// ---------------------------------------------------------
// Start recording
// ---------------------------------------------------------

async function startRecording() {

    if (recording) {
        return;
    }


    if (!checkSensorSupport()) {
        return;
    }


    const permission =
        await requestPermission();

    if (!permission) {
        return;
    }


    // Reset session

    sensorData = [];

    sampleCount = 0;

    startTime = Date.now();

    recording = true;


    // Add event listener

    window.addEventListener(
        "devicemotion",
        handleMotion
    );


    // UI

    statusElement.textContent =
        "Recording";

    startBtn.disabled = true;

    stopBtn.disabled = false;

    downloadBtn.disabled = true;


    // Timer

    timerInterval =
        setInterval(() => {

            if (!startTime) {
                return;
            }

            const elapsed =
                (
                    Date.now() -
                    startTime
                ) / 1000;

            elapsedTimeElement.textContent =
                elapsed.toFixed(1);

        }, 100);
}


// ---------------------------------------------------------
// Stop recording
// ---------------------------------------------------------

function stopRecording() {

    if (!recording) {
        return;
    }


    recording = false;


    window.removeEventListener(
        "devicemotion",
        handleMotion
    );


    clearInterval(
        timerInterval
    );


    timerInterval = null;


    statusElement.textContent =
        "Stopped";


    startBtn.disabled = false;

    stopBtn.disabled = true;


    if (sensorData.length > 0) {

        downloadBtn.disabled = false;

    }


    console.log(
        "Recording completed:",
        sensorData.length,
        "samples"
    );
}


// ---------------------------------------------------------
// CSV escaping
// ---------------------------------------------------------

function escapeCSV(value) {

    const stringValue =
        String(value);

    if (
        stringValue.includes(",") ||
        stringValue.includes('"') ||
        stringValue.includes("\n")
    ) {

        return (
            '"' +
            stringValue.replace(
                /"/g,
                '""'
            ) +
            '"'
        );
    }

    return stringValue;
}


// ---------------------------------------------------------
// Convert data to CSV
// ---------------------------------------------------------

function convertToCSV(data) {

    if (!data.length) {
        return "";
    }


    const columns = [
        "epoch_ms",
        "elapsed_s",
        "exercise",
        "intensity",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
        "interval_ms"
    ];


    const header =
        columns.join(",");


    const rows =
        data.map(row => {

            return columns
                .map(column =>
                    escapeCSV(row[column])
                )
                .join(",");
        });


    return [
        header,
        ...rows
    ].join("\n");
}


// ---------------------------------------------------------
// Download CSV
// ---------------------------------------------------------

function downloadCSV() {

    if (!sensorData.length) {

        alert(
            "No sensor data recorded."
        );

        return;
    }


    const csv =
        convertToCSV(sensorData);


    const blob =
        new Blob(
            [csv],
            {
                type: "text/csv;charset=utf-8;"
            }
        );


    const url =
        URL.createObjectURL(blob);


    const link =
        document.createElement("a");


    const exercise =
        exerciseElement.value;

    const intensity =
        intensityElement.value;


    const timestamp =
        new Date()
            .toISOString()
            .replace(/[:.]/g, "-");


    link.href = url;

    link.download =
        `phone_${exercise}_${intensity}_${timestamp}.csv`;


    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);


    URL.revokeObjectURL(url);


    console.log(
        "CSV downloaded."
    );
}


// ---------------------------------------------------------
// Clear data
// ---------------------------------------------------------

function clearData() {

    if (recording) {

        alert(
            "Stop recording before clearing data."
        );

        return;
    }


    sensorData = [];

    sampleCount = 0;


    sampleCountElement.textContent =
        "0";

    elapsedTimeElement.textContent =
        "0.0";


    statusElement.textContent =
        "Not Started";


    downloadBtn.disabled = true;


    const elements = [
        accXElement,
        accYElement,
        accZElement,
        gyroXElement,
        gyroYElement,
        gyroZElement,
    ];


    elements.forEach(
        element => {
            element.textContent =
                "0.000";
        }
    );
}


// ---------------------------------------------------------
// Button events
// ---------------------------------------------------------

startBtn.addEventListener(
    "click",
    startRecording
);

stopBtn.addEventListener(
    "click",
    stopRecording
);

downloadBtn.addEventListener(
    "click",
    downloadCSV
);

clearBtn.addEventListener(
    "click",
    clearData
);