import 'package:app/models/utils.dart';
import 'package:json_annotation/json_annotation.dart';

part 'corpus.g.dart';

class Corpus {
  final String id;
  final String name;
  final String description;

  Corpus({required this.id, required this.name, required this.description});

  factory Corpus.fromJson(Map<String, dynamic> json) {
    return Corpus(
      id: json['id'],
      name: json['name'],
      description: json['description'],
    );
  }
}

@JsonSerializable()
class CorpusEntry {
  @JsonKey(name: '_id', fromJson: idFromJson)
  final String id;
  @JsonKey(fromJson: idFromJson)
  final String corpus;
  @JsonKey(name: 'entry_type')
  final String entryType;
  @JsonKey(name: 'created_at', fromJson: dateTimeFromJson)
  final DateTime createdAt;
  final String? content;
  final List<Message>? messages;

  CorpusEntry({
    required this.id,
    required this.corpus,
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
