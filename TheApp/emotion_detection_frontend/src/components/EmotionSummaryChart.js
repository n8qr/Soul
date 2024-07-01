import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import axios from 'axios';

// Register the components with Chart.js
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const EmotionSummaryChart = () => {
    const [emotionData, setEmotionData] = useState({});

    useEffect(() => {
        axios.get('http://localhost:5000/api/emotion_summary')
            .then(response => {
                setEmotionData(response.data);
            })
            .catch(error => {
                console.error('Error fetching emotion summary:', error);
            });
    }, []);

    const data = {
        labels: Object.keys(emotionData),
        datasets: [
            {
                label: 'Emotion Count',
                data: Object.values(emotionData),
                backgroundColor: '#e8c5ce',
                borderColor: '#807891',
                borderWidth: 1,
            },
        ],
    };

    const options = {
        scales: {
            y: {
                beginAtZero: true,
            },
        },
        plugins: {
            legend: {
                display: false,
            },
        },
    };

    return (
        <div style={styles.container}>
            <h2 style={styles.title}>Emotion Summary</h2>
            <Bar data={data} options={options} />
        </div>
    );
};

const styles = {
    container: {
        marginTop: '20px',
        backgroundColor: '#fff',
        padding: '20px',
        borderRadius: '10px',
        boxShadow: '0px 0px 10px rgba(0,0,0,0.1)',
        width: '100%',
        maxWidth: '640px',
    },
    title: {
        color: '#807891',
        marginBottom: '20px',
    },
};

export default EmotionSummaryChart;
