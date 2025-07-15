import React, { useState } from "react";
import { View, Text, ScrollView, TouchableOpacity, Alert, StyleSheet } from "react-native";

// Simple Radio Button Component
const RadioButton = ({ label, selected, onPress }) => (
  <TouchableOpacity style={styles.radioContainer} onPress={onPress}>
    <View style={[styles.radioCircle, selected && styles.selectedRadio]} />
    <Text style={styles.radioLabel}>{label}</Text>
  </TouchableOpacity>
);

const Assessment = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const [answers, setAnswers] = useState({});
  const [completedSections, setCompletedSections] = useState([]);

  const assessmentSections = [
    {
      title: "Mental Health (PHQ-9 Sample)",
      description: "Questions about your mood and feelings",
      questions: [
        {
          id: "phq1",
          question: "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
          options: [
            { value: "0", label: "Not at all" },
            { value: "1", label: "Several days" },
            { value: "2", label: "More than half the days" },
            { value: "3", label: "Nearly every day" },
          ],
        },
        {
          id: "phq2",
          question: "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
          options: [
            { value: "0", label: "Not at all" },
            { value: "1", label: "Several days" },
            { value: "2", label: "More than half the days" },
            { value: "3", label: "Nearly every day" },
          ],
        },
      ],
    },
    // Add other sections similarly...
  ];

  const handleAnswerChange = (questionId, value) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value,
    }));
  };

  const handleNextSection = () => {
    const currentQuestions = assessmentSections[currentSection].questions;
    const allAnswered = currentQuestions.every(q => answers[q.id] !== undefined);

    if (!allAnswered) {
      Alert.alert("Incomplete", "Please complete all questions in this section before proceeding.");
      return;
    }

    if (!completedSections.includes(currentSection)) {
      setCompletedSections(prev => [...prev, currentSection]);
    }

    if (currentSection < assessmentSections.length - 1) {
      setCurrentSection(currentSection + 1);
    } else {
      Alert.alert("Assessment Complete", "Your responses will help personalize your MindBridge experience.");
      // Optionally navigate or reset
    }
  };

  const progressPercentage = ((currentSection + 1) / assessmentSections.length) * 100;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>Wellness Assessment</Text>
      <Text style={styles.progressText}>
        Section {currentSection + 1} of {assessmentSections.length} ({Math.round(progressPercentage)}%)
      </Text>

      <View style={styles.progressBarBackground}>
        <View style={[styles.progressBarFill, { width: `${progressPercentage}%` }]} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{assessmentSections[currentSection].title}</Text>
        <Text style={styles.sectionDescription}>{assessmentSections[currentSection].description}</Text>

        {assessmentSections[currentSection].questions.map((question, qIndex) => (
          <View key={question.id} style={styles.questionContainer}>
            <Text style={styles.questionText}>{qIndex + 1}. {question.question}</Text>
            {question.options.map(option => (
              <RadioButton
                key={option.value}
                label={option.label}
                selected={answers[question.id] === option.value}
                onPress={() => handleAnswerChange(question.id, option.value)}
              />
            ))}
          </View>
        ))}

        <TouchableOpacity style={styles.nextButton} onPress={handleNextSection}>
          <Text style={styles.nextButtonText}>
            {currentSection === assessmentSections.length - 1 ? "Complete Assessment" : "Next Section"}
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: "#F0F4FF",
    flexGrow: 1,
  },
  header: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#4C1D95",
  },
  progressText: {
    fontSize: 14,
    color: "#444",
    marginBottom: 8,
  },
  progressBarBackground: {
    height: 10,
    backgroundColor: "#ddd",
    borderRadius: 5,
    overflow: "hidden",
    marginBottom: 20,
  },
  progressBarFill: {
    height: 10,
    backgroundColor: "#7C3AED",
  },
  section: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    marginBottom: 4,
    color: "#312E81",
  },
  sectionDescription: {
    fontSize: 14,
    color: "#555",
    marginBottom: 16,
  },
  questionContainer: {
    marginBottom: 16,
  },
  questionText: {
    fontSize: 16,
    fontWeight: "500",
    marginBottom: 8,
    color: "#1E293B",
  },
  radioContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  radioCircle: {
    height: 20,
    width: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: "#7C3AED",
    marginRight: 12,
  },
  selectedRadio: {
    backgroundColor: "#7C3AED",
  },
  radioLabel: {
    fontSize: 14,
    color: "#334155",
  },
  nextButton: {
    backgroundColor: "#7C3AED",
    paddingVertical: 12,
    borderRadius: 6,
    alignItems: "center",
    marginTop: 24,
  },
  nextButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
});

export default Assessment;
