import classification from '../utils/classification';

export interface IImageFile {
  data: Buffer;
  mimetype: string;
}

class Service {
  private SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif'];

  async classifyImageFile(image: IImageFile) {
    if (!this.SUPPORTED_IMAGE_TYPES.includes(image.mimetype)) {
      throw new Error(
        `Expected image (${this.SUPPORTED_IMAGE_TYPES}), but got ${image.mimetype}`,
      );
    }
    return classification.classify(image.data, 10);
  }
}

export default new Service();
