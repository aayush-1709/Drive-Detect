const NeuralArchitecture = () => {
  return (
    <div className="min-h-screen bg-[#020202] text-white px-6 py-20">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-10 text-center">
          Neural Architecture
        </h1>

        <div className="space-y-10">
          <section>
            <h2 className="text-2xl font-semibold mb-3 text-blue-500">
              Model Architecture
            </h2>
            <p className="text-gray-400 leading-relaxed">
              Drive Detect utilizes a Convolutional Neural Network (CNN)
              designed for image classification tasks. The architecture
              consists of stacked convolutional layers followed by
              pooling layers and fully connected layers for final prediction.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3 text-blue-500">
              CNN Layers Structure
            </h2>
            <p className="text-gray-400 leading-relaxed">
              The model includes multiple convolutional layers for feature
              extraction, ReLU activation for non-linearity, max-pooling
              layers for dimensionality reduction, and dense layers for
              classification.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3 text-blue-500">
              Training Approach
            </h2>
            <p className="text-gray-400 leading-relaxed">
              The model is trained on labeled traffic sign datasets using
              supervised learning. Optimization is performed using gradient
              descent-based algorithms to minimize classification loss.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3 text-blue-500">
              Dataset Overview
            </h2>
            <p className="text-gray-400 leading-relaxed">
              The training dataset includes categorized traffic sign images
              covering regulatory, warning, and mandatory signs to ensure
              robust real-world detection capability.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default NeuralArchitecture;