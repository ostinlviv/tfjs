import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tfnode from '@tensorflow/tfjs-node';

const labels = [
  {
    "id": 0,
    "name": "background"
  },
  {
    "id": 1,
    "name": "vegetables | leafy_greens"
  },
  {
    "id": 2,
    "name": "vegetables | stem_vegetables"
  },
  {
    "id": 3,
    "name": "vegetables | non-starchy_roots"
  },
  {
    "id": 4,
    "name": "vegetables | other"
  },
  {
    "id": 5,
    "name": "fruits"
  },
  {
    "id": 6,
    "name": "protein | meat"
  },
  {
    "id": 7,
    "name": "protein | poultry"
  },
  {
    "id": 8,
    "name": "protein | seafood"
  },
  {
    "id": 9,
    "name": "protein | eggs"
  },
  {
    "id": 10,
    "name": "protein | beans/nuts"
  },
  {
    "id": 11,
    "name": "starches/grains | baked_goods"
  },
  {
    "id": 12,
    "name": "starches/grains | rice/grains/cereals"
  },
  {
    "id": 13,
    "name": "starches/grains | noodles/pasta"
  },
  {
    "id": 14,
    "name": "starches/grains | starchy_vegetables"
  },
  {
    "id": 15,
    "name": "starches/grains | other"
  },
  {
    "id": 16,
    "name": "soups/stews"
  },
  {
    "id": 17,
    "name": "herbs/spices"
  },
  {
    "id": 18,
    "name": "dairy"
  },
  {
    "id": 19,
    "name": "snacks"
  },
  {
    "id": 20,
    "name": "sweets/desserts"
  },
  {
    "id": 21,
    "name": "beverages"
  },
  {
    "id": 22,
    "name": "fats/oils/sauces"
  },
  {
    "id": 23,
    "name": "food_containers"
  },
  {
    "id": 24,
    "name": "dining_tools"
  },
  {
    "id": 25,
    "name": "other_food"
  }
]

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
        className: labels[indicesArr[i]].name,
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
