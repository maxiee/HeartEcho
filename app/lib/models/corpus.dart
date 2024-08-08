import 'package:json_annotation/json_annotation.dart';

part 'corpus.g.dart';

@JsonSerializable()
class Corpus {
  final String id;
  final String name;
  final String description;
  @JsonKey(name: 'created_at')
  final DateTime createdAt;
  @JsonKey(name: 'updated_at')
  final DateTime updatedAt;

  Corpus({
    required this.id,
    required this.name,
    required this.description,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Corpus.fromJson(Map<String, dynamic> json) => _$CorpusFromJson(json);
  Map<String, dynamic> toJson() => _$CorpusToJson(this);
}

@JsonSerializable()
class CorpusEntry {
  final String id;
  final String corpus;
  @JsonKey(name: 'entry_type')
  final String entryType;
  @JsonKey(name: 'created_at')
  final DateTime createdAt;
  final String? content;
  final List<Message>? messages;
  final Map<String, dynamic> metadata;
  final String sha256; // 添加 sha256 字段

  CorpusEntry({
    required this.id,
    required this.corpus,
    required this.entryType,
    required this.createdAt,
    this.content,
    this.messages,
    required this.metadata,
    required this.sha256, // 在构造函数中添加 sha256
  });

  factory CorpusEntry.fromJson(Map<String, dynamic> json) =>
      _$CorpusEntryFromJson(json);
  Map<String, dynamic> toJson() => _$CorpusEntryToJson(this);
}

@JsonSerializable()
class Message {
  final String role;
  final String content;

  Message({required this.role, required this.content});

  factory Message.fromJson(Map<String, dynamic> json) =>
      _$MessageFromJson(json);
  Map<String, dynamic> toJson() => _$MessageToJson(this);
}
