import React, { useState } from 'react';

const TextInput = ({ onSend }) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        onSend(input);
        setInput('');
    };

    return (
        <form onSubmit={handleSubmit} style={{ marginTop: '20px' }}>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type here..."
                style={{ padding: '10px', width: '80%', fontSize: '16px' }}
            />
            <button type="submit" style={{ padding: '10px 20px', fontSize: '16px' }}>Send</button>
        </form>
    );
};

export default TextInput;
