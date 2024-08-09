class TrainingSession {
  final String id;
  final String name;
  final String baseModel;
  final DateTime startTime;
  final DateTime lastTrained;
  final DateTime? endTime;
  final int tokensTrained;
  final Map<String, dynamic> metrics;

  TrainingSession({
    required this.id,
    required this.name,
    required this.baseModel,
    required this.startTime,
    required this.lastTrained,
    this.endTime,
    required this.tokensTrained,
    required this.metrics,
  });

  factory TrainingSession.fromJson(Map<String, dynamic> json) {
    return TrainingSession(
      id: json['id'],
      name: json['name'],
      baseModel: json['base_model'],
      startTime: DateTime.parse(json['start_time']),
      lastTrained: DateTime.parse(json['last_trained']),
      endTime:
          json['end_time'] != null ? DateTime.parse(json['end_time']) : null,
      tokensTrained: json['tokens_trained'] ?? 0,
      metrics: json['metrics'],
    );
  }
}
