import * as tfnode from '@tensorflow/tfjs-node';
import classes from '../classes.json';

class Classification {
  model: tfnode.InferenceModel | undefined;

  async loadModel() {
    if (this.model) {
      return this.model;
    }
    this.model = await tfnode.node.loadSavedModel("./seefood_model", [], 'default');
    return this.model;
  }

  async getTopKClasses(logits: any, topK: any) {
    const {values, indices} = tfnode.topk(logits, topK, true);
    const valuesArr = await values.data();
    const indicesArr = await indices.data();
    const topClassesAndProbs = [];
    for (let i = 0; i < topK; i++) {
      topClassesAndProbs.push({
        className: classes[indicesArr[i]].name,
        probability: valuesArr[i]
      })
    }
    return topClassesAndProbs;
  }

  async classify(imageBuffer: Buffer, topk = 3) {
    if (!this.model) {
      await this.loadModel();
    }
    const uint8array = new Uint8Array(imageBuffer);
    const offset = tfnode.scalar(255);
    const input = tfnode.node.decodeImage(uint8array, 3);
    const formattedImg = input
      .resizeNearestNeighbor([513, 513])
      .div(offset)
      .toFloat()
      .expandDims();
    console.log('formattedImg', formattedImg)
    const output = this.model?.predict(formattedImg, {}) as any;
    console.log('output', output)
    const result = await this.getTopKClasses(output[1], topk)
    return result;
  }
}

export default new Classification();
