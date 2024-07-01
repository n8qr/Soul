import React from 'react';

const AnalysisChoicePopup = ({ onSelect }) => {
    return (
        <div style={popupStyle}>
            <h2>Choose Analysis Method</h2>
            <button onClick={() => onSelect('video')} style={buttonStyle}>Video Analysis</button>
            <button onClick={() => onSelect('audio')} style={buttonStyle}>Audio Analysis</button>
            <button onClick={() => onSelect('text')} style={buttonStyle}>Text Analysis</button>
        </div>
    );
};

const popupStyle = {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    backgroundColor: '#fff',
    padding: '20px',
    borderRadius: '10px',
    boxShadow: '0px 0px 10px rgba(0,0,0,0.1)',
    zIndex: 1000
};

const buttonStyle = {
    display: 'block',
    width: '100%',
    padding: '10px',
    marginTop: '10px',
    fontSize: '16px',
    backgroundColor: '#e8c5ce',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer'
};

export default AnalysisChoicePopup;
