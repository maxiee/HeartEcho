import 'package:json_annotation/json_annotation.dart';

part 'corpus.g.dart';

class Corpus {
  final String id;
  final String name;
  final String description;

  Corpus({required this.id, required this.name, required this.description});

  factory Corpus.fromJson(Map<String, dynamic> json) {
    return Corpus(
      id: json['_id']['\$oid'],
      name: json['name'],
      description: json['description'],
    );
  }
}

@JsonSerializable()
class CorpusEntry {
  final String id;
  final String corpusId;
  final String entryType;
  final DateTime createdAt;
  final String? content;
  final List<Message>? messages;

  CorpusEntry({
    required this.id,
    required this.corpusId,
    required this.entryType,
    required this.createdAt,
    this.content,
    this.messages,
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
