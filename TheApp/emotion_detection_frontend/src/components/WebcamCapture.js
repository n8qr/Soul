import React, { useEffect, useRef } from 'react';

const WebcamCapture = () => {
    const videoRef = useRef(null);

    useEffect(() => {
        const startVideo = () => {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    let video = videoRef.current;
                    if (video) {
                        video.srcObject = stream;
                        video.play();
                    }
                })
                .catch(err => {
                    console.error("Error accessing webcam: ", err);
                });
        };
        startVideo();
    }, []);

    return (
        <video
            ref={videoRef}
            style={{ width: '100%', height: '100%', borderRadius: '10px' }}
            autoPlay
            playsInline
        />
    );
};

export default WebcamCapture;
