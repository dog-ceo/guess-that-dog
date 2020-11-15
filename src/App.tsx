import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const breeds = ['affenpinscher', 'african', 'airedale', 'akita', 'appenzeller', 'australian-shepherd', 'basenji', 'beagle', 'bluetick', 'borzoi', 'bouvier', 'boxer', 'brabancon', 'briard', 'buhund-norwegian', 'bulldog-boston', 'bulldog-english', 'bulldog-french', 'bullterrier-staffordshire', 'cairn', 'cattledog-australian', 'chihuahua', 'chow', 'clumber', 'cockapoo', 'collie', 'collie-border', 'coonhound', 'corgi', 'corgi-cardigan', 'cotondetulear', 'dachshund', 'dalmatian', 'dane-great', 'deerhound-scottish', 'dhole', 'dingo', 'doberman', 'elkhound-norwegian', 'entlebucher', 'eskimo', 'finnish-lapphund', 'frise-bichon', 'germanshepherd', 'greyhound', 'greyhound-italian', 'groenendael', 'havanese', 'hound-afghan', 'hound-basset', 'hound-blood', 'hound-english', 'hound-ibizan', 'hound-plott', 'hound-walker', 'husky', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'labrador', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese', 'mastiff-bull', 'mastiff-english', 'mastiff-tibetan', 'mexicanhairless', 'mix', 'mountain-bernese', 'mountain-swiss', 'newfoundland', 'otterhound', 'ovcharka-caucasian', 'papillon', 'pekinese', 'pembroke', 'pinscher-miniature', 'pitbull', 'pointer-german', 'pointer-germanlonghair', 'pomeranian', 'poodle-miniature', 'poodle-standard', 'poodle-toy', 'pug', 'puggle', 'pyrenees', 'redbone', 'retriever-chesapeake', 'retriever-curly', 'retriever-flatcoated', 'retriever-golden', 'ridgeback-rhodesian', 'rottweiler', 'saluki', 'samoyed', 'schipperke', 'schnauzer', 'schnauzer-giant', 'schnauzer-miniature', 'setter-english', 'setter-gordon', 'setter-irish', 'sheepdog-english', 'sheepdog-shetland', 'shiba', 'shihtzu', 'spaniel-blenheim', 'spaniel-brittany', 'spaniel-cocker', 'spaniel-irish', 'spaniel-japanese', 'spaniel-sussex', 'spaniel-welsh', 'springer-english', 'stbernard', 'terrier-american', 'terrier-australian', 'terrier-bedlington', 'terrier-border', 'terrier-dandie', 'terrier-fox', 'terrier-irish', 'terrier-kerryblue', 'terrier-lakeland', 'terrier-norfolk', 'terrier-norwich', 'terrier-patterdale', 'terrier-russell', 'terrier-scottish', 'terrier-sealyham', 'terrier-silky', 'terrier-tibetan', 'terrier-toy', 'terrier-westhighland', 'terrier-wheaten', 'terrier-yorkshire', 'vizsla', 'waterdog-spanish', 'weimaraner', 'whippet', 'wolfhound-irish']
  const fileInput = React.useRef<HTMLInputElement>();

  async function handleClick() {
    fileInput.current.click()
  }

  async function handleChange(event: any) {
    document.getElementById('result').innerHTML = "Ummmm.....";

    const model = await tf.loadGraphModel('model/1/model.json');
    const reader = new FileReader();
    const image = document.getElementById("img") as HTMLImageElement;

    reader.onload = async (event) => {
      image.src = event.target.result as string;

      image.onload = async () => {
        const inputTensor = tf.browser.fromPixels(image);
        const resized = tf.image.resizeBilinear(inputTensor, [224, 224]).toFloat();
        const offset = tf.scalar(255.0);
        const normalized = resized.div(offset);
        const batched = normalized.expandDims(0);

        const outputTensor = await model.predict(batched) as tf.Tensor;

        const prediction = await outputTensor.data();

        let guess = Array.from(prediction).map(function(p, i) {
          return {
            probability: p,
            breed: breeds[i - 1]
          };
        }).sort(function(a,b){
          return b.probability-a.probability;
        }).slice(0,1)[0];

        console.log(guess.probability);

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

        document.getElementById('result').innerHTML = `So I'm ${word} sure that is a ${guess.breed}`
      }
    };

    reader.readAsDataURL(event.target.files[0]);
  }

  return (
    <div>
      <div className="main">
        <div className="controls">
          <input type='file' ref={fileInput} onChange={handleChange} />
          <button id='upload' onClick={handleClick} >Upload a dog image</button><br />
          <div id="result"></div>
        </div>
        <img id="img"/>
      </div>
    </div>
  );
}

export default App;
