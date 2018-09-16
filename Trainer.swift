#!/usr/bin/swift
// CreateML trainer

import CreateML
import Foundation

// Specify where is the data
let path = NSString(string: "~/dataset/").expandingTildeInPath
let trainingDir = URL(fileURLWithPath: path).appendingPathComponent("Training")
let testDir = URL(fileURLWithPath: path).appendingPathComponent("Test")

// Create a mode
let model = try MLImageClassifier(trainingData: .labeledDirectories(at: trainingDir))

// Test the model
let evaluation = model.evaluation(on: .labeledDirectories(at: testDir))

// Save the model
let home = NSString(string: "~").expandingTildeInPath
try model.write(to: URL(fileURLWithPath: home).appendingPathComponent("AnimalClassifier.mlmodel"))
