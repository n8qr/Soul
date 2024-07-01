import React, { useState, useRef } from 'react';
import axios from 'axios';

const AudioRecorder = ({ onDetect }) => {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const startRecording = () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                const mediaRecorder = new MediaRecorder(stream);
                mediaRecorderRef.current = mediaRecorder;
                mediaRecorder.start();
                setIsRecording(true);

                mediaRecorder.ondataavailable = event => {
                    audioChunksRef.current.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                    audioChunksRef.current = [];
                    const formData = new FormData();
                    formData.append('audio', audioBlob);

                    try {
                        const response = await axios.post('http://localhost:5000/api/audio_emotion', formData, {
                            headers: {
                                'Content-Type': 'multipart/form-data'
                            }
                        });
                        onDetect(response.data.response);
                    } catch (error) {
                        console.error('Error sending audio data to backend:', error);
                    }
                };

                // Automatically stop after 10 seconds
                setTimeout(() => {
                    if (mediaRecorder.state === "recording") {
                        mediaRecorder.stop();
                        setIsRecording(false);
                    }
                }, 10000); // 10 seconds
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
            });
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return (
        <div>
            {isRecording ? (
                <button onClick={stopRecording} style={buttonStyle}>Stop Recording</button>
            ) : (
                <button onClick={startRecording} style={buttonStyle}>Start Recording</button>
            )}
        </div>
    );
};

const buttonStyle = {
    padding: '10px 20px',
    fontSize: '16px',
    backgroundColor: '#e8c5ce',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    marginTop: '20px'
};

export default AudioRecorder;
