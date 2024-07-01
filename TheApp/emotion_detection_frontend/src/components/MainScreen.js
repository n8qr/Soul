import React, { useState, useEffect } from 'react';
import VideoStream from './VideoStream';
import AudioRecorder from './AudioRecorder';
import TextInput from './TextInput';
import AnalysisChoicePopup from './AnalysisChoicePopup';
import axios from 'axios';

const MainScreen = () => {
    const [messages, setMessages] = useState([]);
    const [emotionDetected, setEmotionDetected] = useState(false);
    const [started, setStarted] = useState(false);
    const [analysisMethod, setAnalysisMethod] = useState(null);

    useEffect(() => {
        if (analysisMethod !== 'video') {
            axios.post('http://localhost:5000/stop_video_analysis')
                .then(response => {
                    console.log(response.data.message);
                })
                .catch(error => {
                    console.error("Error stopping video analysis:", error);
                });
        }
    }, [analysisMethod]);

    const handleSend = (message) => {
        setMessages([...messages, { sender: 'user', text: message }]);
        axios.post('http://localhost:5000/api/message', { message })
            .then(response => {
                const gptResponse = response.data.response;
                setMessages(prevMessages => [...prevMessages, { sender: 'bot', text: gptResponse }]);
                if (gptResponse.includes("Conversation ended")) {
                    setEmotionDetected(false);
                    setStarted(false);
                }
            })
            .catch(error => {
                console.error("Error sending message to backend:", error);
            });
    };

    const handleDetect = (message) => {
        setMessages([{ sender: 'bot', text: message }]);
        setEmotionDetected(true);
        speak(message);
    };

    const handleTextAnalysis = (message) => {
        axios.post('http://localhost:5000/api/text_emotion', { text: message })
            .then(response => {
                const gptResponse = response.data.response;
                setMessages([{ sender: 'bot', text: gptResponse }]);
                setEmotionDetected(true);
                speak(gptResponse);
            })
            .catch(error => {
                console.error("Error analyzing text:", error);
            });
    };

    const handleStart = () => {
        axios.post('http://localhost:5000/api/start')
            .then(response => {
                const gptResponse = response.data.response;
                setMessages([{ sender: 'bot', text: gptResponse }]);
                setStarted(true);
                speak(gptResponse);
            })
            .catch(error => {
                console.error("Error starting conversation:", error);
            });
    };

    const handleRestart = () => {
        setMessages([]);
        setEmotionDetected(false);
        setStarted(false);
        setAnalysisMethod(null);
    };

    const speak = (text) => {
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        const femaleVoice = voices.find(voice => voice.name.includes('Female') || voice.name.includes('Samantha') || voice.name.includes('Google US English'));
        if (femaleVoice) {
            utterance.voice = femaleVoice;
        }
        utterance.rate = 1;
        window.speechSynthesis.speak(utterance);
    };

    useEffect(() => {
        const loadVoices = () => {
            if (window.speechSynthesis.getVoices().length > 0) {
                speak('');
            } else {
                window.speechSynthesis.onvoiceschanged = () => speak('');
            }
        };
        loadVoices();
    }, []);

    return (
        <div style={{ textAlign: 'center', padding: '20px', backgroundColor: '#e8c5ce', height: '100vh', borderRadius: '20px', overflowY: 'auto' }}>
            <h1 style={{ color: '#807891' }}>Joy</h1>
            <p style={{ color: '#807891' }}>Your happiness, our mission.</p>
            {!analysisMethod && (
                <AnalysisChoicePopup onSelect={setAnalysisMethod} />
            )}
            {analysisMethod === 'video' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
                    <div style={{ width: '100%', maxWidth: '640px', backgroundColor: '#fff', padding: '20px', borderRadius: '20px', boxShadow: '0px 0px 10px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                        <VideoStream />
                    </div>
                    <div style={{ width: '100%', maxWidth: '640px', backgroundColor: '#fff', padding: '20px', borderRadius: '20px', boxShadow: '0px 0px 10px rgba(0,0,0,0.1)', height: '300px', overflowY: 'auto', marginBottom: '20px' }}>
                        <div>
                            {messages.map((message, index) => (
                                <div key={index} style={{ textAlign: message.sender === 'user' ? 'right' : 'left', color: message.sender === 'user' ? '#625c70' : '#807891' }}>
                                    <p>{message.text}</p>
                                    {message.sender === 'bot' && (
                                        <button onClick={() => speak(message.text)} style={{ padding: '5px 10px', fontSize: '14px', marginTop: '5px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                            Play Audio
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>
                        {started ? (
                            <TextInput onSend={handleSend} />
                        ) : (
                            <button onClick={handleStart} style={{ padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                Start Conversation
                            </button>
                        )}
                        <button onClick={handleRestart} style={{ marginTop: '20px', padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                            Restart Conversation
                        </button>
                    </div>
                </div>
            )}
            {analysisMethod === 'audio' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
                    <AudioRecorder onDetect={handleDetect} />
                    <div style={{ width: '100%', maxWidth: '640px', backgroundColor: '#fff', padding: '20px', borderRadius: '20px', boxShadow: '0px 0px 10px rgba(0,0,0,0.1)', height: '300px', overflowY: 'auto', marginBottom: '20px' }}>
                        <div>
                            {messages.map((message, index) => (
                                <div key={index} style={{ textAlign: message.sender === 'user' ? 'right' : 'left', color: message.sender === 'user' ? '#625c70' : '#807891' }}>
                                    <p>{message.text}</p>
                                    {message.sender === 'bot' && (
                                        <button onClick={() => speak(message.text)} style={{ padding: '5px 10px', fontSize: '14px', marginTop: '5px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                            Play Audio
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>
                        {started ? (
                            <TextInput onSend={handleSend} />
                        ) : (
                            <button onClick={handleStart} style={{ padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                Start Conversation
                            </button>
                        )}
                        <button onClick={handleRestart} style={{ marginTop: '20px', padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                            Restart Conversation
                        </button>
                    </div>
                </div>
            )}
            {analysisMethod === 'text' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
                    <TextInput onSend={handleTextAnalysis} />
                    <div style={{ width: '100%', maxWidth: '640px', backgroundColor: '#fff', padding: '20px', borderRadius: '20px', boxShadow: '0px 0px 10px rgba(0,0,0,0.1)', height: '300px', overflowY: 'auto', marginBottom: '20px' }}>
                        <div>
                            {messages.map((message, index) => (
                                <div key={index} style={{ textAlign: message.sender === 'user' ? 'right' : 'left', color: message.sender === 'user' ? '#625c70' : '#807891' }}>
                                    <p>{message.text}</p>
                                    {message.sender === 'bot' && (
                                        <button onClick={() => speak(message.text)} style={{ padding: '5px 10px', fontSize: '14px', marginTop: '5px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                            Play Audio
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>
                        {started ? (
                            <TextInput onSend={handleSend} />
                        ) : (
                            <button onClick={handleStart} style={{ padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                Start Conversation
                            </button>
                        )}
                        <button onClick={handleRestart} style={{ marginTop: '20px', padding: '10px 20px', fontSize: '16px', backgroundColor: '#e8c5ce', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                            Restart Conversation
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MainScreen;
