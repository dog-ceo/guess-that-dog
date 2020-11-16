import React, { ChangeEvent, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import breeds from './lib/breeds';
import './App.css';

function App() {
  const [imgSrc, setImgSrc] = useState('');
  const [message, setMessage] = useState('');
  const fileInput = React.useRef<HTMLInputElement>();

  async function handleClick() {
    fileInput.current.click()
  }

  async function handleChange(event: ChangeEvent<HTMLInputElement>) {
    setMessage('Ummmm.....');

    const model = await tf.loadGraphModel('model/1/model.json');
    const reader = new FileReader();
    const image = document.getElementById("img") as HTMLImageElement;

    reader.onload = async (event) => {
      setImgSrc(event.target.result as string);

      document.getElementById('main').style.backgroundImage = `url(${ event.target.result })`;

      image.onload = async () => {
        const inputTensor = tf.browser.fromPixels(image);
        const resized = tf.image.resizeBilinear(inputTensor, [224, 224]).toFloat();
        const offset = tf.scalar(255.0);
        const normalized = resized.div(offset);
        const batched = normalized.expandDims(0);

        const outputTensor = await model.predict(batched) as tf.Tensor;
        const prediction = await outputTensor.data();

        let guess = Array.from(prediction).map((probability, index) => ({
          probability,
          breed: breeds[index - 1]
        })).sort((a,b) => b.probability-a.probability).slice(0,1)[0];

        const confidence = Math.round(guess.probability * 100);

        let word;

        if (confidence >= 90) {
          word = 'pretty'
        } else if (confidence >= 80) {
          word = 'quite'
        } else if (confidence >= 60) {
          word = 'kinda'
        } else if (confidence >= 40) {
          word = 'vaguely'
        } else if (confidence >= 20) {
          word = 'only slightly'
        } else {
          word = 'only a little bit'
        }

        setMessage(`So I'm ${word} sure that is a ${guess.breed}?`);
      }
    };

    reader.readAsDataURL(event.target.files[0]);
  }

  return (
    <div>
      <div id="main">
        <div className="controls">
          <input type='file' ref={fileInput} onChange={handleChange} />
          <button id='upload' onClick={handleClick} >Submit a dog image</button>
          <div id="result">{message}</div>
        </div>
        <img alt="dog" id="img" src={imgSrc}/>
      </div>
    </div>
  );
}

export default App;
