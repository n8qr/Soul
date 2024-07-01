import React, { useState } from 'react';
import MainScreen from './components/MainScreen';
import WelcomeScreen from './components/WelcomeScreen';

const App = () => {
    const [started, setStarted] = useState(false);

    const handleStart = () => {
        setStarted(true);
    };

    return (
        <div>
            {started ? <MainScreen /> : <WelcomeScreen onStart={handleStart} />}
        </div>
    );
};

export default App;
