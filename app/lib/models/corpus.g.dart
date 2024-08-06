// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'corpus.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

CorpusEntry _$CorpusEntryFromJson(Map<String, dynamic> json) => CorpusEntry(
      id: json['id'] as String,
      corpusId: json['corpusId'] as String,
      entryType: json['entryType'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
      content: json['content'] as String?,
      messages: (json['messages'] as List<dynamic>?)
          ?.map((e) => Message.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$CorpusEntryToJson(CorpusEntry instance) =>
    <String, dynamic>{
      'id': instance.id,
      'corpusId': instance.corpusId,
      'entryType': instance.entryType,
      'createdAt': instance.createdAt.toIso8601String(),
      'content': instance.content,
      'messages': instance.messages,
    };

Message _$MessageFromJson(Map<String, dynamic> json) => Message(
      role: json['role'] as String,
      content: json['content'] as String,
    );

Map<String, dynamic> _$MessageToJson(Message instance) => <String, dynamic>{
      'role': instance.role,
      'content': instance.content,
    };
