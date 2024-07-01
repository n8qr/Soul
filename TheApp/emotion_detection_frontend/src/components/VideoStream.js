import React from 'react';

const VideoStream = () => {
    return (
        <div>
            <img
                src="http://localhost:5000/video_feed"
                alt="Video Feed"
                style={{ width: '100%', borderRadius: '10px' }}
            />
        </div>
    );
};

export default VideoStream;
