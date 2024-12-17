let normalizationData = null;
let allow = 1
let loadingImage = document.querySelector(".loading-image-container")
const MeantempValue = document.querySelector(".meantemp_container")

const seasonalSelect = document.querySelector(".seasonal-input")
const columnsSelect = document.querySelector(".columns-input")

const dailyContainer = document.querySelector(".single-column-inputs")
const weeklyContainer = document.querySelector(".multiple-column-inputs")


async function predict(inputFeatures, sequence ,path, key, model) {

    const session = await ort.InferenceSession.create(path);

    let tensor = ""

    if (model == "ensemble"){
      tensor = new ort.Tensor(
        'float32',
        new Float32Array(inputFeatures), 
        [1, inputFeatures.length] 
      )
    }
    else {
    tensor = new ort.Tensor(
      'float32',
      new Float32Array(inputFeatures), 
      [1, sequence, inputFeatures.length / sequence] 
    )
    }

    const feeds = {};
    feeds[key] = tensor

    try {
      const result = await session.run(feeds);
      return result;
  } catch (err) {
      console.error("Error running the model : ", err);
      throw err;
  }

    return result;
}

async function loadNormalizationInfo() {
  if (normalizationData) {
    return normalizationData;
  }

  const response = await fetch('./models/normalization_info.json');
  normalizationData = await response.json(); 
  return normalizationData;
}

async function normalizeInputs(inputs) {
  const normalizationJsonInfo = await loadNormalizationInfo(); 

  let result = { status: true, message: '', data: [] };

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];

    if (input.value == null || input.value == undefined || input.value === '') {
      result.status = false;
      result.message = 'The ' + input.getAttribute('placeholder') + ' is empty.';
      break;
    } 
    else if (input.value < 0){
      result.status = false;
      result.message = 'The ' + input.getAttribute('placeholder') + ' cannot be less than 0.';
      break;
    }

    let input_min_value = normalizationJsonInfo.min_values[input.getAttribute("data-key")]
    let input_max_value = normalizationJsonInfo.max_values[input.getAttribute("data-key")]

    let normalizeInput = (input.value - input_min_value) / (input_max_value - input_min_value);
    result.data.push(normalizeInput)
  }

  return result;
}

function formatValue(value) {

  if (value.includes('.')) {
    return false;
  }

  const valueStr = String(value).replace(/,/g, ''); 

  if (valueStr.length > 3) {
    return `${valueStr.slice(0, -3)}.${valueStr.slice(-3)}`;
  } else {
    return `0.${valueStr.padStart(3, '0')}`;
  }
}

async function deNormalizeValue(normalizedValue, target) {

    const normalizationJsonInfo = await loadNormalizationInfo();
  
    let value_min_value = normalizationJsonInfo.min_values[target]
    let value_max_value = normalizationJsonInfo.max_values[target]
    
    value = (normalizedValue * (value_max_value - value_min_value)) + value_min_value;

    return Math.round(value)
  }

document.querySelector('.calculate').addEventListener('click', async () => {

  if (allow == 1){
    
  allow = 0  
  loadingImage.classList.remove("hidden")
  MeantempValue.classList.add("hidden")

  let values = []

  if (columnsSelect.value == "one-column"){
    values = document.querySelectorAll('.single-column-inputs input')
  }
  else {
    values = document.querySelectorAll('.multiple-column-inputs input')
  }

  const normalizeValues = await normalizeInputs(values); 

  if(!normalizeValues.status){
    allow = 1 
    loadingImage.classList.add("hidden")
    return window.alert(normalizeValues.message);
  }

  const algorithm = document.querySelector(".algorithm-input").value
  let deNormalizePrediction = 0

  if (algorithm == "neural_network"){

    modelPath = ""

    if (columnsSelect.value == "one-column" && seasonalSelect.value == "1"){
      modelPath = "./models/deep_learning_daily_model.onnx"
    }
    else if(columnsSelect.value == "multiple-columns" && seasonalSelect.value == "1"){
      modelPath = "./models/deep_learning_daily_x_model.onnx"
    }
    else if (columnsSelect.value == "one-column" && seasonalSelect.value == "7"){
      modelPath = "./models/deep_learning_weekly_model.onnx"
    }
    else {
      modelPath = "./models/deep_learning_weekly_x_model.onnx"
    }

    let normalizePrediction = await predict(normalizeValues.data, parseInt(seasonalSelect.value), modelPath, "input", "deep learning")
    deNormalizePrediction = await deNormalizeValue(normalizePrediction.output.data[0] , "meantemp")
  }
  else {

    if (columnsSelect.value == "one-column" && seasonalSelect.value == "1"){
      modelPath = "./models/ensemble_daily_model.onnx"
    }
    else if(columnsSelect.value == "multiple-columns" && seasonalSelect.value == "1"){
      modelPath = "./models/ensemble_daily_x_model.onnx"
    }
    else if (columnsSelect.value == "one-column" && seasonalSelect.value == "7"){
      modelPath = "./models/ensemble_weekly_model.onnx"
    }
    else {
      modelPath = "./models/ensemble_weekly_x_model.onnx"
    }

    let normalizePrediction = await predict(normalizeValues.data,parseInt(seasonalSelect.value), modelPath, "float_input", "ensemble")
    deNormalizePrediction = await deNormalizeValue(normalizePrediction.variable.cpuData[0] , "meantemp")
  }

  loadingImage.classList.add("hidden")
  MeantempValue.classList.remove("hidden")
  document.querySelector("#meantemp_value").innerHTML = Math.abs(deNormalizePrediction)
  allow = 1
  
}
});

document.querySelector(".algorithm-input").addEventListener("input", (e)=>{

  const warningText = document.querySelector(".warning-text")

   if(e.target.value == "ensembles"){
    warningText.classList.remove("hidden")
   }
   else {
    warningText.classList.add("hidden")
   }

})

seasonalSelect.addEventListener("input", (e)=>{
  setForecastInputs()
})

columnsSelect.addEventListener("input", (e)=>{

  if (e.target.value == "one-column"){
    dailyContainer.classList.remove("none")
    weeklyContainer.classList.add("none")
  }
  else {
    dailyContainer.classList.add("none")
    weeklyContainer.classList.remove("none")
  }

  setForecastInputs()

})

function setForecastInputs() {

  let seasonalInputs = document.querySelectorAll(".input-daily")
  let weeklyInputs = document.querySelectorAll(".input-weekly")
  
  let inputs = [];
  let container = "";

  if (columnsSelect.value == "one-column") {
    inputs = seasonalInputs;
    container = dailyContainer;
  } else {
    inputs = weeklyInputs;
    container = weeklyContainer;
  }

  let copyInput = inputs[0].outerHTML; 

  if (inputs.length < seasonalSelect.value) {
    for (let i = 1; i < seasonalSelect.value; i++) {
      const newNode = document.createRange().createContextualFragment(copyInput).firstElementChild;

      if (columnsSelect.value == "one-column") {
        newNode.placeholder = newNode.placeholder.replace("1", i + 1) 
      }
      else {
        let temp = newNode.querySelector(".m-c-temp")
        let hum = newNode.querySelector(".m-c-hum")
        let wind = newNode.querySelector(".m-c-wind")
        let press = newNode.querySelector(".m-c-press")

        temp.placeholder = temp.placeholder.replace("1", i + 1) 
        hum.placeholder = hum.placeholder.replace("1", i + 1) 
        wind.placeholder = wind.placeholder.replace("1", i + 1) 
        press.placeholder = press.placeholder.replace("1", i + 1) 
      }

      container.appendChild(newNode);
      
    }
  }
  else if(inputs.length > seasonalSelect.value){
    while (container.children.length > 1) {
      container.removeChild(container.lastChild);
    }

  }

}