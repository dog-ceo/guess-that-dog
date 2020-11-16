import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const breeds = [
    'affenpinscher',
    'african',
    'airedale',
    'akita',
    'appenzeller',
    'australian shepherd',
    'basenji',
    'beagle',
    'bluetick',
    'borzoi',
    'bouvier',
    'boxer',
    'brabancon',
    'briard',
    'norwegian buhund',
    'boston bulldog',
    'english bulldog',
    'french bulldog',
    'staffie',
    'cairn',
    'australian cattledog',
    'chihuahua',
    'chow',
    'clumber',
    'cockapoo',
    'collie',
    'border collie',
    'coonhound',
    'corgi',
    'cardigan corgi',
    'cotondetulear',
    'dachshund',
    'dalmatian',
    'great dane',
    'scottish deerhound',
    'dhole',
    'dingo',
    'doberman',
    'norwegian elkhound',
    'entlebucher',
    'eskimo',
    'finnish lapphund',
    'frise bichon',
    'german shepherd',
    'greyhound',
    'italian greyhound',
    'groenendael',
    'havanese',
    'afghan hound',
    'basset hound',
    'blood hound',
    'english hound',
    'ibizan hound',
    'plott hound',
    'walker hound',
    'husky',
    'keeshond',
    'kelpie',
    'komondor',
    'kuvasz',
    'labrador',
    'leonberg',
    'lhasa',
    'malamute',
    'malinois',
    'maltese',
    'bull mastiff',
    'english mastiff',
    'mtibetan astiff',
    'mexican hairless',
    'mixed breed',
    'bernese mountain dog',
    'swiss mountain dog',
    'newfoundland',
    'otterhound',
    'caucasian ovcharka',
    'papillon',
    'pekinese',
    'pembroke',
    'miniature pinscher',
    'pitbull',
    'german pointer',
    'longhair german pointer',
    'pomeranian',
    'miniature poodle',
    'poodle',
    'toy poodle',
    'pug',
    'puggle',
    'pyrenees',
    'redbone',
    'chesapeake retriever',
    'curly retriever',
    'flatcoated retriever',
    'golden retriever',
    'rhodesian ridgeback',
    'rottweiler',
    'saluki',
    'samoyed',
    'schipperke',
    'schnauzer',
    'giant schnauzer',
    'miniature schnauzer',
    'english setter',
    'gordon setter',
    'irish setter',
    'english sheepdog',
    'shetland sheepdog',
    'shiba',
    'shihtzu',
    'blenheim spaniel',
    'brittany spaniel',
    'cocker spaniel',
    'irish spaniel',
    'japanese spaniel',
    'sussex spaniel',
    'welsh spaniel',
    'english springer',
    'saint bernard',
    'american terrier',
    'australian terrier',
    'bedlington terrier',
    'border terrier',
    'dandie terrier',
    'fox terrier',
    'irish terrier',
    'kerryblue terrier',
    'lakeland terrier',
    'norfolk terrier',
    'norwich terrier',
    'patterdale terrier',
    'russell terrier',
    'scottish terrier',
    'sealyham terrier',
    'silky terrier',
    'tibetan terrier',
    'toy terrier',
    'westhighland terrier',
    'wheaten terrier',
    'yorkshire terrier',
    'vizsla',
    'spanish waterdog',
    'weimaraner',
    'whippet',
    'irish wolfhound'
  ]
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


      document.getElementById('main').style.backgroundImage = `url(${ event.target.result })`;

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

        document.getElementById('result').innerHTML = `So I'm ${word} sure that is a ${guess.breed}?`
      }
    };

    reader.readAsDataURL(event.target.files[0]);
  }

  return (
    <div>
      <div id="main">
        <div className="controls">
          <input type='file' ref={fileInput} onChange={handleChange} />
          <button id='upload' onClick={handleClick} >Submit a dog image</button><br />
          <div id="result"></div>
        </div>
        <img id="img"/>
      </div>
    </div>
  );
}

export default App;
