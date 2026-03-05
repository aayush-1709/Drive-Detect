import { useState } from "react";
import { ChevronDown } from "lucide-react";

const faqData = [
  {
    question: "What is DriveDetect?",
    answer:
      "DriveDetect is an AI-powered traffic sign recognition system that uses a deep learning model to classify traffic signs from uploaded images.",
  },
  {
    question: "How does DriveDetect work?",
    answer:
      "When a user uploads an image, the system processes it using a trained convolutional neural network (CNN) that identifies patterns and predicts the traffic sign category with a confidence score.",
  },
  {
    question: "What dataset is used for training?",
    answer:
      "DriveDetect is trained on traffic sign datasets containing labeled images of regulatory, warning, and mandatory road signs to ensure accurate recognition.",
  },
  {
    question: "What image formats are supported?",
    answer:
      "Users can upload SVG, PNG, JPG, or GIF images with a maximum file size of 5MB.",
  },
  {
    question: "Is my uploaded image stored?",
    answer:
      "No. Uploaded images are processed temporarily for inference and are not permanently stored on the server.",
  },
  {
    question: "Can I run DriveDetect locally?",
    answer:
      "Yes. You can clone the GitHub repository, install the required dependencies, and run both the frontend and backend locally for development and testing.",
  },
  {
    question: "How accurate is the model?",
    answer:
      "The model achieves high classification accuracy thanks to its optimized convolutional neural network architecture trained on thousands of labeled traffic sign images.",
  },
  {
    question: "Is DriveDetect open source?",
    answer:
      "Yes. DriveDetect is an open-source project and contributions from developers are welcome through GitHub pull requests.",
  },
  {
    question: "Where can I report bugs or suggest improvements?",
    answer:
      "You can open an issue on the GitHub repository to report bugs, request features, or contribute improvements to the project.",
  },
];

export function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section
      id="faq"
      className="py-24 px-6 bg-gray-50 dark:bg-gray-950 transition-colors duration-300"
    >
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
          Frequently <span className="text-blue-600">Asked Questions</span>
        </h2>

        <div className="space-y-4">
          {faqData.map((item, index) => (
            <div
              key={index}
              className="border border-gray-200 dark:border-white/10 rounded-xl bg-white dark:bg-white/5 transition"
            >
              <button
                onClick={() => toggle(index)}
                className="w-full flex justify-between items-center p-6 text-left"
              >
                <span className="font-medium">{item.question}</span>
                <ChevronDown
                  className={`transition-transform duration-300 ${
                    openIndex === index ? "rotate-180" : ""
                  }`}
                />
              </button>

              {openIndex === index && (
                <div className="px-6 pb-6 text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  {item.answer}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}