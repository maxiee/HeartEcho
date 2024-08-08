// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'corpus.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Corpus _$CorpusFromJson(Map<String, dynamic> json) => Corpus(
      id: json['id'] as String,
      name: json['name'] as String,
      description: json['description'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      updatedAt: DateTime.parse(json['updated_at'] as String),
    );

Map<String, dynamic> _$CorpusToJson(Corpus instance) => <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'description': instance.description,
      'created_at': instance.createdAt.toIso8601String(),
      'updated_at': instance.updatedAt.toIso8601String(),
    };

CorpusEntry _$CorpusEntryFromJson(Map<String, dynamic> json) => CorpusEntry(
      id: json['id'] as String,
      corpus: json['corpus'] as String,
      entryType: json['entry_type'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      content: json['content'] as String?,
      messages: (json['messages'] as List<dynamic>?)
          ?.map((e) => Message.fromJson(e as Map<String, dynamic>))
          .toList(),
      metadata: json['metadata'] as Map<String, dynamic>,
      sha256: json['sha256'] as String,
    );

Map<String, dynamic> _$CorpusEntryToJson(CorpusEntry instance) =>
    <String, dynamic>{
      'id': instance.id,
      'corpus': instance.corpus,
      'entry_type': instance.entryType,
      'created_at': instance.createdAt.toIso8601String(),
      'content': instance.content,
      'messages': instance.messages,
      'metadata': instance.metadata,
      'sha256': instance.sha256,
    };

Message _$MessageFromJson(Map<String, dynamic> json) => Message(
      role: json['role'] as String,
      content: json['content'] as String,
    );

Map<String, dynamic> _$MessageToJson(Message instance) => <String, dynamic>{
      'role': instance.role,
      'content': instance.content,
    };
