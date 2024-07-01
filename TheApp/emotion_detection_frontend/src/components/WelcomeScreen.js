import React, { useState } from 'react';
import axios from 'axios';
import EmotionSummaryChart from './EmotionSummaryChart';

const WelcomeScreen = ({ onStart }) => {
    const [audio] = useState(new Audio('/Joy.mp3'));

    const handlePlayAudio = () => {
        audio.play();
    };

    return (
        <div style={styles.container}>
            <img src='/Joy.png' alt="Joy Logo" style={styles.logo} />
            <button style={styles.button} onClick={onStart}>Get Started</button>
            <button style={styles.audioButton} onClick={handlePlayAudio}>Play Instructions</button>
            <EmotionSummaryChart />
        </div>
    );
};

const styles = {
    container: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        backgroundColor: '#e8c5ce',
        color: '#625c70',
        textAlign: 'center',
        flexDirection: 'column',
    },
    logo: {
        maxWidth: '1000px',
        marginBottom: '50px',
    },
    button: {
        marginTop: '20px',
        padding: '20px 40px',
        fontSize: '16px',
        backgroundColor: '#f4e7e1',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        color: '#625c70',
        outline: 'none',
    },
    audioButton: {
        marginTop: '10px',
        padding: '20px 40px',
        fontSize: '16px',
        backgroundColor: '#f4e7e1',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        color: '#625c70',
        outline: 'none',
    },
};

export default WelcomeScreen;
