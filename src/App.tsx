import React, { ChangeEvent, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { GraphModel } from '@tensorflow/tfjs';
import breeds from './lib/breeds';
import './App.css';

function App() {
  const [model, setModel] = useState<GraphModel>();
  const [imgSrc, setImgSrc] = useState('');
  const [message, setMessage] = useState('');
  const fileInput = React.createRef<HTMLInputElement>();

  useEffect(() => {
    (async () => {
      setModel(await tf.loadGraphModel('model/1/model.json'));
    })();
  }, []);

  function handleClick() {
    fileInput.current!.click();
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    setMessage('Ummmm.....');

    const reader = new FileReader();

    reader.onload = (event) => {
      setImgSrc(event.target!.result as string);
    };

    if (event.target.files) {
      reader.readAsDataURL(event.target.files[0]);
    }
  }

  async function handleLoad(event: ChangeEvent<HTMLImageElement>) {
    const image = event.target;
    const inputTensor = tf.browser.fromPixels(image);
    const resized = tf.image.resizeBilinear(inputTensor, [224, 224]).toFloat();
    const offset = tf.scalar(255.0);
    const normalized = resized.div(offset);
    const batched = normalized.expandDims(0);

    const outputTensor = await model!.predict(batched) as tf.Tensor;
    const prediction = await outputTensor.data();

    const guesses = Array.from(prediction).map((probability, index) => ({
      probability,
      breed: breeds[index - 1]
    })).sort((a,b) => b.probability - a.probability);

    const bestGuess = guesses[0];
    const nextBestGuess = guesses[1];
    const confidence = bestGuess.probability;

    let confidenceWord;

    if (confidence >= 0.9) {
      confidenceWord = 'pretty'
    } else if (confidence >= 0.8) {
      confidenceWord = 'quite'
    } else if (confidence >= 0.6) {
      confidenceWord = 'kinda'
    } else if (confidence >= 0.4) {
      confidenceWord = 'vaguely'
    } else if (confidence >= 0.2) {
      confidenceWord = 'only slightly'
    } else {
      confidenceWord = 'only a little bit'
    }

    let message = `So I'm ${confidenceWord} sure that is a ${bestGuess.breed}`;

    if (confidence < 0.5) {
      message = `${message} or maybe a ${nextBestGuess.breed}`;
    }

    setMessage(`${message} ?`);
  }

  return (
    <div>
      <div id="main" style={{ backgroundImage: `url(${imgSrc})` }}>
        <div className="controls">
          <input type='file' ref={fileInput} onChange={handleChange} accept="image/*" />
          <button id='upload' onClick={handleClick} >Submit a dog image</button>
          <div id="result">{message}</div>
        </div>
        <img alt="dog" id="img" src={imgSrc} onLoad={handleLoad} />
      </div>
    </div>
  );
}

export default App;
