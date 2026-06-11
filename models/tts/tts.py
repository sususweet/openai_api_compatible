from typing import Mapping, Optional

from dify_plugin.entities.model import AIModelEntity, I18nObject

from dify_plugin.interfaces.model.openai_compatible.tts import OAICompatText2SpeechModel

from models.common_openai import auth_requests_session, log_validation_auth_config


class OpenAIText2SpeechModel(OAICompatText2SpeechModel):
    def validate_credentials(self, model: str, credentials: dict) -> None:
        log_validation_auth_config(credentials)
        with auth_requests_session(credentials, log_requests=True):
            super().validate_credentials(model, credentials)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        content_text: str,
        user: Optional[str] = None,
    ):
        with auth_requests_session(credentials):
            return super()._invoke(model, credentials, content_text, user=user)

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_us=credentials["display_name"], zh_hans=credentials["display_name"]
            )

        return entity
